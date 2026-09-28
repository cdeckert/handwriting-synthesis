import Darwin
import Foundation
import HandwritingCore

private struct CLIError: LocalizedError {
    let message: String
    var errorDescription: String? { message }
}

private struct Arguments {
    var text: String?
    var inputPath: String?
    var outputPath = "handwriting.pdf"
    var format: OutputFormat?
    var options = GenerationOptions()
    var pngScale = 2.0
    var listStyles = false
    var showHelp = false

    init(_ raw: [String]) throws {
        var index = 0
        func value(after option: String) throws -> String {
            guard index + 1 < raw.count else { throw CLIError(message: "Missing value after \(option).") }
            index += 1
            return raw[index]
        }

        while index < raw.count {
            let argument = raw[index]
            switch argument {
            case "-h", "--help": showHelp = true
            case "--list-styles": listStyles = true
            case "-t", "--text": text = try value(after: argument)
            case "-i", "--input": inputPath = try value(after: argument)
            case "-o", "--output": outputPath = try value(after: argument)
            case "--format":
                let rawValue = try value(after: argument).lowercased()
                guard let parsed = OutputFormat(rawValue: rawValue) else {
                    throw CLIError(message: "Unknown format '\(rawValue)'. Use pdf, png, or svg.")
                }
                format = parsed
            case "--engine":
                let rawValue = try value(after: argument).lowercased()
                guard let parsed = GenerationEngine(rawValue: rawValue) else {
                    throw CLIError(message: "Engine must be auto, neural, or unicode.")
                }
                options.engine = parsed
            case "--font": options.unicodeFont = try value(after: argument)
            case "--style":
                guard let parsed = Int(try value(after: argument)), (0 ... 12).contains(parsed) else {
                    throw CLIError(message: "Style must be a number from 0 through 12.")
                }
                options.style = parsed
            case "--bias":
                guard let parsed = Double(try value(after: argument)), (0.1 ... 1.5).contains(parsed) else {
                    throw CLIError(message: "Bias must be between 0.1 and 1.5.")
                }
                options.bias = parsed
            case "--page-size":
                let rawValue = try value(after: argument).lowercased()
                guard let parsed = PageFormat(rawValue: rawValue) else {
                    throw CLIError(message: "Page size must be a5, a4, a3, letter, or legal.")
                }
                options.pageFormat = parsed
            case "--orientation":
                let rawValue = try value(after: argument).lowercased()
                guard let parsed = PageOrientation(rawValue: rawValue) else {
                    throw CLIError(message: "Orientation must be portrait or landscape.")
                }
                options.pageOrientation = parsed
            case "--font-size":
                guard let parsed = Double(try value(after: argument)), (18 ... 72).contains(parsed) else {
                    throw CLIError(message: "Font size must be between 18 and 72 points.")
                }
                options.fontSize = parsed
            case "--alignment":
                let rawValue = try value(after: argument).lowercased()
                guard let parsed = TextAlignment(rawValue: rawValue) else {
                    throw CLIError(message: "Alignment must be left or center.")
                }
                options.alignment = parsed
            case "--ink": options.inkColor = try value(after: argument)
            case "--background": options.backgroundColor = try value(after: argument)
            case "--seed":
                guard let parsed = UInt64(try value(after: argument)) else {
                    throw CLIError(message: "Seed must be an unsigned integer.")
                }
                options.seed = parsed
            case "--png-scale":
                guard let parsed = Double(try value(after: argument)), (0.5 ... 4).contains(parsed) else {
                    throw CLIError(message: "PNG scale must be between 0.5 and 4.")
                }
                pngScale = parsed
            default:
                throw CLIError(message: "Unknown option '\(argument)'.")
            }
            index += 1
        }
    }

    func inputText() throws -> String {
        if let text { return text }
        if let inputPath {
            if inputPath == "-" {
                return String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
            }
            return try String(contentsOfFile: inputPath, encoding: .utf8)
        }
        if isatty(STDIN_FILENO) == 0 {
            return String(decoding: FileHandle.standardInput.readDataToEndOfFile(), as: UTF8.self)
        }
        throw CLIError(message: "Provide text with --text, --input, or stdin.")
    }

    func resolvedFormat() throws -> OutputFormat {
        if let format { return format }
        let pathExtension = URL(fileURLWithPath: outputPath).pathExtension.lowercased()
        guard let inferred = OutputFormat(rawValue: pathExtension) else {
            throw CLIError(message: "Cannot infer output format. Use --format pdf, png, or svg.")
        }
        return inferred
    }
}

private struct SuccessReport: Encodable {
    let format: String
    let engine: String
    let pages: Int
    let files: [String]
    let normalizedText: String
}

private let help = """
handwriting-cli - create handwritten pages locally with Apple Core ML

USAGE
  handwriting-cli --text "Hello" --output note.pdf
  handwriting-cli --input letter.txt --output letter.pdf
  cat letter.txt | handwriting-cli --input - --output pages.png

OPTIONS
  -t, --text TEXT            Text to render
  -i, --input PATH           UTF-8 text file, or - for stdin
  -o, --output PATH          Output path (default: handwriting.pdf)
      --format FORMAT        pdf, png, or svg; inferred from output extension
      --engine ENGINE        auto, neural, or unicode (default: auto)
      --font NAME            Unicode handwriting font (default: Noteworthy-Light)
      --style NUMBER         Handwriting style 0...12 (default: 9)
      --bias NUMBER          Neatness 0.1...1.5 (default: 0.75)
      --page-size SIZE       a5, a4, a3, letter, or legal (default: a4)
      --orientation VALUE    portrait or landscape (default: portrait)
      --font-size POINTS     18...72 (default: 36)
      --alignment VALUE      left or center (default: left)
      --ink COLOR            Ink as #RRGGBB (default: #172554)
      --background COLOR     Paper as #RRGGBB (default: #FFFFFF)
      --seed NUMBER          Reproducible random seed
      --png-scale NUMBER     PNG resolution multiplier 0.5...4 (default: 2)
      --list-styles          Print all bundled styles
  -h, --help                 Show this help

Long input is wrapped and split across as many pages as required. In auto mode,
the neural engine handles its original alphabet and the Unicode handwriting
engine handles umlauts, accents, symbols, and other scripts without changing
the text. All generation stays on this Mac.
"""

private func writeError(_ message: String) {
    FileHandle.standardError.write(Data((message + "\n").utf8))
}

do {
    let arguments = try Arguments(Array(CommandLine.arguments.dropFirst()))
    if arguments.showHelp {
        print(help)
        exit(EXIT_SUCCESS)
    }
    if arguments.listStyles {
        for style in HandwritingStyle.bundled {
            print("\(style.id)\t\(style.label)\t\(style.detail)")
        }
        exit(EXIT_SUCCESS)
    }

    let text = try arguments.inputText()
    let format = try arguments.resolvedFormat()
    let outputURL = URL(fileURLWithPath: arguments.outputPath).standardizedFileURL
    let generator = try HandwritingGenerator()
    var lastProgress = -1
    let generated = try generator.generate(text: text, options: arguments.options) { value in
        let percent = Int(value * 100)
        if percent >= lastProgress + 10 || percent == 100 {
            writeError("Generating: \(percent)%")
            lastProgress = percent
        }
    }
    let files = try HandwritingRenderer.write(
        generated,
        format: format,
        to: outputURL,
        pngScale: arguments.pngScale
    )
    let report = SuccessReport(
        format: format.rawValue,
        engine: generated.engine.rawValue,
        pages: generated.pages.count,
        files: files.map(\.path),
        normalizedText: generated.normalizedText
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    print(String(decoding: try encoder.encode(report), as: UTF8.self))
} catch {
    writeError("Error: \(error.localizedDescription)")
    exit(EXIT_FAILURE)
}
