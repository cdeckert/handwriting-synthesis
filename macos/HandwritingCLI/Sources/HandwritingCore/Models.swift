import Foundation

public enum HandwritingError: LocalizedError, Equatable {
    case emptyText
    case textTooLong(Int)
    case unsupportedCharacter(Character)
    case missingStyle(Int)
    case invalidStyleData(Int)
    case invalidModelOutput
    case invalidColor(String)
    case outputCreationFailed(String)

    public var errorDescription: String? {
        switch self {
        case .emptyText:
            return "The input text is empty."
        case let .textTooLong(limit):
            return "The input exceeds the limit of \(limit) characters."
        case let .unsupportedCharacter(character):
            return "The character '\(character)' is not supported by the handwriting model."
        case let .missingStyle(style):
            return "The bundled data for style \(style) is missing."
        case let .invalidStyleData(style):
            return "The bundled data for style \(style) is unreadable."
        case .invalidModelOutput:
            return "Core ML returned invalid handwriting data."
        case let .invalidColor(color):
            return "'\(color)' is not a valid #RRGGBB color."
        case let .outputCreationFailed(path):
            return "Could not create output at \(path)."
        }
    }
}

public struct HandwritingStyle: Identifiable, Equatable, Codable, Sendable {
    public let id: Int
    public let label: String
    public let detail: String

    public static let bundled = [
        HandwritingStyle(id: 0, label: "Light Loops", detail: "Thin and relaxed"),
        HandwritingStyle(id: 1, label: "Wide Print", detail: "Open and geometric"),
        HandwritingStyle(id: 2, label: "Compact Script", detail: "Small and connected"),
        HandwritingStyle(id: 3, label: "Tall Flourish", detail: "High, expressive strokes"),
        HandwritingStyle(id: 4, label: "Rounded Notes", detail: "Soft and spacious"),
        HandwritingStyle(id: 5, label: "Fine Print", detail: "Neat and understated"),
        HandwritingStyle(id: 6, label: "Casual Loops", detail: "Loose and personal"),
        HandwritingStyle(id: 7, label: "Open Round", detail: "Clear and generous"),
        HandwritingStyle(id: 8, label: "Elegant Narrow", detail: "Tall and refined"),
        HandwritingStyle(id: 9, label: "Bold Slant", detail: "Confident and fluid"),
        HandwritingStyle(id: 10, label: "Playful Script", detail: "Bouncy and informal"),
        HandwritingStyle(id: 11, label: "Fine Cursive", detail: "Small and flowing"),
        HandwritingStyle(id: 12, label: "Airy Script", detail: "Light and spacious"),
    ]
}

public enum TextAlignment: String, CaseIterable, Codable, Sendable {
    case left
    case center
}

public enum PageOrientation: String, CaseIterable, Codable, Sendable {
    case portrait
    case landscape
}

public enum PageFormat: String, CaseIterable, Codable, Sendable {
    case a5
    case a4
    case a3
    case letter
    case legal

    public func dimensions(orientation: PageOrientation) -> PageDimensions {
        let natural: PageDimensions = switch self {
        case .a5: PageDimensions(width: 419.53, height: 595.28)
        case .a4: PageDimensions(width: 595.28, height: 841.89)
        case .a3: PageDimensions(width: 841.89, height: 1190.55)
        case .letter: PageDimensions(width: 612, height: 792)
        case .legal: PageDimensions(width: 612, height: 1008)
        }
        let shortEdge = min(natural.width, natural.height)
        let longEdge = max(natural.width, natural.height)
        switch orientation {
        case .portrait:
            return PageDimensions(width: shortEdge, height: longEdge)
        case .landscape:
            return PageDimensions(width: longEdge, height: shortEdge)
        }
    }
}

public struct PageDimensions: Codable, Equatable, Sendable {
    public let width: Double
    public let height: Double

    public init(width: Double, height: Double) {
        self.width = width
        self.height = height
    }
}

public struct GenerationOptions: Equatable, Sendable {
    public var engine: GenerationEngine
    public var style: Int
    public var bias: Double
    public var alignment: TextAlignment
    public var pageFormat: PageFormat
    public var pageOrientation: PageOrientation
    public var fontSize: Double
    public var inkColor: String
    public var backgroundColor: String
    public var unicodeFont: String
    public var seed: UInt64?

    public init(
        engine: GenerationEngine = .auto,
        style: Int = 9,
        bias: Double = 0.75,
        alignment: TextAlignment = .left,
        pageFormat: PageFormat = .a4,
        pageOrientation: PageOrientation = .portrait,
        fontSize: Double = 36,
        inkColor: String = "#172554",
        backgroundColor: String = "#FFFFFF",
        unicodeFont: String = "Noteworthy-Light",
        seed: UInt64? = nil
    ) {
        self.engine = engine
        self.style = style
        self.bias = bias
        self.alignment = alignment
        self.pageFormat = pageFormat
        self.pageOrientation = pageOrientation
        self.fontSize = fontSize
        self.inkColor = inkColor
        self.backgroundColor = backgroundColor
        self.unicodeFont = unicodeFont
        self.seed = seed
    }
}

public struct RenderPoint: Codable, Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let move: Bool
}

public struct RenderedPath: Codable, Equatable, Sendable {
    public let strokeColor: String
    public let lineWidth: Double
    public let points: [RenderPoint]
}

public struct RenderedGlyph: Codable, Equatable, Sendable {
    public let text: String
    public let x: Double
    public let baselineY: Double
    public let fontName: String
    public let fontSize: Double
    public let rotation: Double
    public let color: String

    public init(
        text: String,
        x: Double,
        baselineY: Double,
        fontName: String,
        fontSize: Double,
        rotation: Double,
        color: String
    ) {
        self.text = text
        self.x = x
        self.baselineY = baselineY
        self.fontName = fontName
        self.fontSize = fontSize
        self.rotation = rotation
        self.color = color
    }
}

public struct RenderDocument: Codable, Equatable, Sendable {
    public let width: Double
    public let height: Double
    public let backgroundColor: String
    public let paths: [RenderedPath]
    public let glyphs: [RenderedGlyph]

    public init(
        width: Double,
        height: Double,
        backgroundColor: String,
        paths: [RenderedPath] = [],
        glyphs: [RenderedGlyph] = []
    ) {
        self.width = width
        self.height = height
        self.backgroundColor = backgroundColor
        self.paths = paths
        self.glyphs = glyphs
    }
}

public struct GeneratedDocument: Equatable, Sendable {
    public let pages: [RenderDocument]
    public let normalizedText: String
    public let engine: GenerationEngine

    public init(
        pages: [RenderDocument],
        normalizedText: String,
        engine: GenerationEngine
    ) {
        self.pages = pages
        self.normalizedText = normalizedText
        self.engine = engine
    }
}

public enum OutputFormat: String, CaseIterable, Sendable {
    case pdf
    case png
    case svg
}

public enum GenerationEngine: String, CaseIterable, Codable, Sendable {
    case auto
    case neural
    case unicode
}

struct HandwritingPageLayout: Equatable {
    let dimensions: PageDimensions
    let fontSize: Double
    let margin: Double
    let lineHeight: Double
    let contentWidth: Double
    let maximumCharactersPerLine: Int
    let maximumLines: Int

    init(format: PageFormat, orientation: PageOrientation, fontSize: Double) {
        dimensions = format.dimensions(orientation: orientation)
        self.fontSize = min(max(fontSize, 18), 72)
        margin = max(24, min(54, min(dimensions.width, dimensions.height) * 0.075))
        lineHeight = self.fontSize * 1.55
        contentWidth = max(dimensions.width - (margin * 2), self.fontSize * 2)
        let contentHeight = max(dimensions.height - (margin * 2), lineHeight)
        maximumCharactersPerLine = min(
            HandwritingGenerator.maximumCharactersPerLine,
            max(4, Int(floor(contentWidth / (self.fontSize * 0.34))))
        )
        maximumLines = max(1, Int(floor(contentHeight / lineHeight)))
    }
}

enum HandwritingTextLayouter {
    static func wrap(_ text: String, maximumCharactersPerLine: Int) -> [String] {
        let limit = max(1, maximumCharactersPerLine)
        let paragraphs = text.split(separator: "\n", omittingEmptySubsequences: false).map(String.init)
        var lines = [String]()

        for paragraph in paragraphs {
            guard !paragraph.isEmpty else {
                lines.append("")
                continue
            }

            let words = paragraph.split(separator: " ").map(String.init)
            guard !words.isEmpty else {
                lines.append("")
                continue
            }

            var currentLine = ""
            for word in words {
                if word.count > limit {
                    if !currentLine.isEmpty {
                        lines.append(currentLine)
                        currentLine = ""
                    }
                    var remaining = Array(word)
                    while remaining.count > limit {
                        lines.append(String(remaining.prefix(limit)))
                        remaining.removeFirst(limit)
                    }
                    currentLine = String(remaining)
                    continue
                }

                let candidate = currentLine.isEmpty ? word : currentLine + " " + word
                if candidate.count <= limit {
                    currentLine = candidate
                } else {
                    lines.append(currentLine)
                    currentLine = word
                }
            }
            if !currentLine.isEmpty {
                lines.append(currentLine)
            }
        }
        return lines
    }
}
