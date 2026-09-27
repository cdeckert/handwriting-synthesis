import CoreML
import Foundation

enum NativeHandwritingError: LocalizedError {
    case emptyText
    case textTooLong
    case lineTooLong(Int)
    case contentDoesNotFit(required: Int, available: Int)
    case unsupportedCharacter(Character)
    case missingStyle(Int)
    case invalidStyleData(Int)
    case invalidModelOutput

    var errorDescription: String? {
        switch self {
        case .emptyText:
            "Enter some text to generate handwriting."
        case .textTooLong:
            "Use no more than 911 characters."
        case let .lineTooLong(line):
            "Line \(line) is longer than 75 characters."
        case let .contentDoesNotFit(required, available):
            "This text needs \(required) lines, but the selected page and writing size fit \(available). Choose a larger page, a smaller writing size, or landscape orientation."
        case let .unsupportedCharacter(character):
            "The character ‘\(character)’ is not supported by this model."
        case let .missingStyle(style):
            "The bundled data for style \(style + 1) is missing."
        case let .invalidStyleData(style):
            "The bundled data for style \(style + 1) is unreadable."
        case .invalidModelOutput:
            "Core ML returned invalid handwriting data."
        }
    }
}

actor HandwritingGenerationService {
    private var generator: NativeHandwritingGenerator?

    func render(
        request: GenerationRequest,
        bias: Double,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) throws -> RenderDocument {
        if generator == nil {
            generator = try NativeHandwritingGenerator()
        }
        return try generator!.render(
            request: request,
            bias: bias,
            progress: progress
        )
    }
}

final class NativeHandwritingGenerator {
    static let maximumTextLength = 911
    static let maximumCharactersPerLine = 75

    private static let alphabet: [Character] = [
        "\0", " ", "!", "\"", "#", "'", "(", ")", ",", "-", ".",
        "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ":", ";",
        "?", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K",
        "L", "M", "N", "O", "P", "R", "S", "T", "U", "V", "W", "Y",
        "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l",
        "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x",
        "y", "z",
    ]

    private static let characterNumbers = Dictionary(
        uniqueKeysWithValues: alphabet.enumerated().map { ($1, Int32($0)) }
    )

    private let model: HandwritingStep
    private let styleStore = HandwritingStyleStore()

    init() throws {
        let configuration = MLModelConfiguration()
        configuration.computeUnits = .all
        model = try HandwritingStep(configuration: configuration)
    }

    func render(
        request: GenerationRequest,
        bias: Double,
        progress: @escaping @Sendable (Double) -> Void = { _ in }
    ) throws -> RenderDocument {
        let pageLayout = HandwritingPageLayout(
            format: request.pageFormat,
            orientation: request.pageOrientation,
            fontSize: request.fontSize
        )
        let lines = try validatedLines(request.text, pageLayout: pageLayout)
        let styleID = request.style ?? 9
        let style = try styleStore.load(id: styleID)
        let totalSteps = lines.reduce(0) { total, line in
            guard !line.isEmpty else { return total }
            return total + style.strokes.count + max(line.count * 40, 40)
        }
        var progressReporter = GenerationProgressReporter(
            total: max(totalSteps, 1),
            callback: progress
        )
        progressReporter.start()
        var random = GaussianRandomSource()
        var sampledLines = [[StrokeOffset]?]()
        sampledLines.reserveCapacity(lines.count)

        for line in lines {
            try Task.checkCancellation()
            if line.isEmpty {
                sampledLines.append(nil)
            } else {
                sampledLines.append(
                    try sample(
                        line: line,
                        style: style,
                        bias: Float(bias),
                        random: &random,
                        progress: &progressReporter
                    )
                )
            }
        }

        progressReporter.finish()

        return Self.layout(
            sampledLines: sampledLines,
            alignment: request.alignment,
            pageLayout: pageLayout,
            unit: request.pageFormat.isPaper ? .points : .pixels
        )
    }

    private func validatedLines(
        _ text: String,
        pageLayout: HandwritingPageLayout
    ) throws -> [String] {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw NativeHandwritingError.emptyText
        }

        let normalized = text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
        guard normalized.count <= Self.maximumTextLength else {
            throw NativeHandwritingError.textTooLong
        }
        for character in normalized where character != "\n" {
            if Self.characterNumbers[character] == nil {
                throw NativeHandwritingError.unsupportedCharacter(character)
            }
        }

        let lines = HandwritingTextLayouter.wrap(
            normalized,
            maximumCharactersPerLine: pageLayout.maximumCharactersPerLine
        )
        guard lines.count <= pageLayout.maximumLines else {
            throw NativeHandwritingError.contentDoesNotFit(
                required: lines.count,
                available: pageLayout.maximumLines
            )
        }
        return lines
    }

    private func sample(
        line: String,
        style: HandwritingStyleSample,
        bias: Float,
        random: inout GaussianRandomSource,
        progress: inout GenerationProgressReporter
    ) throws -> [StrokeOffset] {
        let encoded = try encode(style.characters + " " + line)
        var state = RecurrentState()
        var lastParameters: [Float]?

        for stroke in style.strokes {
            try Task.checkCancellation()
            let output = try step(stroke: stroke, encoded: encoded, state: state)
            state = output.state
            lastParameters = output.gmmParameters
            progress.advance()
        }

        guard let primedParameters = lastParameters else {
            throw NativeHandwritingError.invalidModelOutput
        }

        var previousStroke = try Self.sampleStroke(
            parameters: primedParameters,
            bias: bias,
            random: &random
        )
        var result = [StrokeOffset]()
        result.reserveCapacity(max(line.count * 30, 80))

        let maximumSteps = max(line.count * 40, 40)
        for _ in 0 ..< maximumSteps {
            try Task.checkCancellation()
            let output = try step(
                stroke: previousStroke,
                encoded: encoded,
                state: state
            )
            state = output.state
            let sampled = try Self.sampleStroke(
                parameters: output.gmmParameters,
                bias: bias,
                random: &random
            )
            result.append(sampled)
            previousStroke = sampled
            progress.advance()

            let characterIndex = state.phi.indices.max {
                state.phi[$0] < state.phi[$1]
            } ?? 0
            let finalCharacter = characterIndex >= encoded.length - 1
            let pastFinalCharacter = characterIndex >= encoded.length
            if pastFinalCharacter || (finalCharacter && sampled.penUp == 1) {
                break
            }
        }

        guard !result.isEmpty else {
            throw NativeHandwritingError.invalidModelOutput
        }
        return result
    }

    private func encode(_ text: String) throws -> EncodedText {
        var values = [Int32]()
        values.reserveCapacity(text.count + 1)
        for character in text {
            guard let value = Self.characterNumbers[character] else {
                throw NativeHandwritingError.unsupportedCharacter(character)
            }
            values.append(value)
        }
        values.append(0)

        guard values.count <= 120 else {
            throw NativeHandwritingError.lineTooLong(1)
        }
        let length = values.count
        values.append(contentsOf: repeatElement(0, count: 120 - values.count))
        return EncodedText(values: values, length: length)
    }

    private func step(
        stroke: StrokeOffset,
        encoded: EncodedText,
        state: RecurrentState
    ) throws -> StepResult {
        let output = try model.prediction(
            stroke: MLShapedArray(scalars: [stroke.x, stroke.y, stroke.penUp], shape: [1, 3]),
            chars: MLShapedArray(scalars: encoded.values, shape: [1, 120]),
            chars_len: MLShapedArray(scalars: [Int32(encoded.length)], shape: [1]),
            h1: state.shaped(state.h1),
            c1: state.shaped(state.c1),
            h2: state.shaped(state.h2),
            c2: state.shaped(state.c2),
            h3: state.shaped(state.h3),
            c3: state.shaped(state.c3),
            kappa: MLShapedArray(scalars: state.kappa, shape: [1, 10]),
            window: MLShapedArray(scalars: state.window, shape: [1, 73])
        )

        let parameters = Array(output.gmm_paramsShapedArray.scalars)
        guard parameters.count == 121, parameters.allSatisfy(\.isFinite) else {
            throw NativeHandwritingError.invalidModelOutput
        }
        return StepResult(
            gmmParameters: parameters,
            state: RecurrentState(output: output)
        )
    }

    private static func sampleStroke(
        parameters: [Float],
        bias: Float,
        random: inout GaussianRandomSource
    ) throws -> StrokeOffset {
        guard parameters.count == 121 else {
            throw NativeHandwritingError.invalidModelOutput
        }

        let scaledLogits = parameters[0 ..< 20].map { Double($0 * (1 + bias)) }
        let largestLogit = scaledLogits.max() ?? 0
        var weights = scaledLogits.map { exp($0 - largestLogit) }
        let unfilteredTotal = weights.reduce(0, +)
        guard unfilteredTotal.isFinite, unfilteredTotal > 0 else {
            throw NativeHandwritingError.invalidModelOutput
        }
        weights = weights.map {
            let probability = $0 / unfilteredTotal
            return probability < 0.01 ? 0 : probability
        }
        let component = random.weightedIndex(weights)

        let sigmaX = exp(Double(parameters[20 + component] - bias))
        let sigmaY = exp(Double(parameters[40 + component] - bias))
        let rho = max(-0.999_999_99, min(0.999_999_99, tanh(Double(parameters[60 + component]))))
        let meanX = Double(parameters[80 + component])
        let meanY = Double(parameters[100 + component])
        let normals = random.normalPair()

        let x = meanX + sigmaX * normals.0
        let y = meanY + sigmaY * (
            rho * normals.0 + sqrt(max(0, 1 - (rho * rho))) * normals.1
        )
        let rawEndProbability = sigmoid(Double(parameters[120]))
        let endProbability = rawEndProbability < 0.01 ? 0 : rawEndProbability
        let penUp: Float = random.unitInterval() < endProbability ? 1 : 0

        guard x.isFinite, y.isFinite else {
            throw NativeHandwritingError.invalidModelOutput
        }
        return StrokeOffset(x: Float(x), y: Float(y), penUp: penUp)
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 {
            return 1 / (1 + exp(-value))
        }
        let exponential = exp(value)
        return exponential / (1 + exponential)
    }
}

private struct EncodedText {
    let values: [Int32]
    let length: Int
}

private struct StepResult {
    let gmmParameters: [Float]
    let state: RecurrentState
}

private struct RecurrentState {
    var h1 = [Float](repeating: 0, count: 400)
    var c1 = [Float](repeating: 0, count: 400)
    var h2 = [Float](repeating: 0, count: 400)
    var c2 = [Float](repeating: 0, count: 400)
    var h3 = [Float](repeating: 0, count: 400)
    var c3 = [Float](repeating: 0, count: 400)
    var kappa = [Float](repeating: 0, count: 10)
    var window = [Float](repeating: 0, count: 73)
    var phi = [Float](repeating: 0, count: 120)

    init() {}

    init(output: HandwritingStepOutput) {
        h1 = Array(output.next_h1ShapedArray.scalars)
        c1 = Array(output.next_c1ShapedArray.scalars)
        h2 = Array(output.next_h2ShapedArray.scalars)
        c2 = Array(output.next_c2ShapedArray.scalars)
        h3 = Array(output.next_h3ShapedArray.scalars)
        c3 = Array(output.next_c3ShapedArray.scalars)
        kappa = Array(output.next_kappaShapedArray.scalars)
        window = Array(output.next_windowShapedArray.scalars)
        phi = Array(output.next_phiShapedArray.scalars)
    }

    func shaped(_ values: [Float]) -> MLShapedArray<Float> {
        MLShapedArray(scalars: values, shape: [1, 400])
    }
}

struct StrokeOffset {
    let x: Float
    let y: Float
    let penUp: Float
}

private struct GaussianRandomSource {
    private var generator = SystemRandomNumberGenerator()

    mutating func unitInterval() -> Double {
        Double.random(in: 0 ..< 1, using: &generator)
    }

    mutating func weightedIndex(_ weights: [Double]) -> Int {
        let total = weights.reduce(0, +)
        guard total > 0 else {
            return weights.indices.max(by: { weights[$0] < weights[$1] }) ?? 0
        }
        var cursor = unitInterval() * total
        for (index, weight) in weights.enumerated() {
            cursor -= weight
            if cursor <= 0 {
                return index
            }
        }
        return weights.indices.last ?? 0
    }

    mutating func normalPair() -> (Double, Double) {
        let first = max(unitInterval(), Double.leastNonzeroMagnitude)
        let second = unitInterval()
        let magnitude = sqrt(-2 * log(first))
        return (
            magnitude * cos(2 * .pi * second),
            magnitude * sin(2 * .pi * second)
        )
    }
}

private struct GenerationProgressReporter {
    let total: Int
    let callback: @Sendable (Double) -> Void
    private var completed = 0
    private var lastReported = -1.0

    init(total: Int, callback: @escaping @Sendable (Double) -> Void) {
        self.total = total
        self.callback = callback
    }

    mutating func start() {
        callback(0)
        lastReported = 0
    }

    mutating func advance() {
        completed += 1
        let value = min(Double(completed) / Double(total), 0.99)
        if value - lastReported >= 0.01 {
            callback(value)
            lastReported = value
        }
    }

    func finish() {
        callback(1)
    }
}

struct HandwritingStyleSample {
    let characters: String
    let strokes: [StrokeOffset]
}

final class HandwritingStyleStore {
    private var cache = [Int: HandwritingStyleSample]()

    func load(id: Int) throws -> HandwritingStyleSample {
        if let cached = cache[id] {
            return cached
        }
        guard (0 ... 12).contains(id) else {
            throw NativeHandwritingError.missingStyle(id)
        }
        guard
            let strokeURL = resourceURL(name: "style-\(id)-strokes"),
            let characterURL = resourceURL(name: "style-\(id)-chars")
        else {
            throw NativeHandwritingError.missingStyle(id)
        }

        let strokes = try Self.decodeStrokes(Data(contentsOf: strokeURL), style: id)
        let characters = try Self.decodeCharacters(
            Data(contentsOf: characterURL),
            style: id
        )
        let sample = HandwritingStyleSample(
            characters: characters,
            strokes: strokes
        )
        cache[id] = sample
        return sample
    }

    private func resourceURL(name: String) -> URL? {
        Bundle.main.url(forResource: name, withExtension: "npy", subdirectory: "Styles")
            ?? Bundle.main.url(forResource: name, withExtension: "npy")
    }

    private static func decodeStrokes(
        _ data: Data,
        style: Int
    ) throws -> [StrokeOffset] {
        let parsed = try parse(data, style: style)
        guard parsed.header.contains("'<f4'"), parsed.payload.count.isMultiple(of: 12) else {
            throw NativeHandwritingError.invalidStyleData(style)
        }

        return parsed.payload.withUnsafeBytes { bytes in
            (0 ..< (parsed.payload.count / 12)).map { row in
                let offset = row * 12
                return StrokeOffset(
                    x: float32(bytes, offset: offset),
                    y: float32(bytes, offset: offset + 4),
                    penUp: float32(bytes, offset: offset + 8)
                )
            }
        }
    }

    private static func decodeCharacters(
        _ data: Data,
        style: Int
    ) throws -> String {
        let parsed = try parse(data, style: style)
        guard parsed.header.contains("'|S") else {
            throw NativeHandwritingError.invalidStyleData(style)
        }
        let bytes = parsed.payload.prefix { $0 != 0 }
        guard let value = String(data: bytes, encoding: .utf8) else {
            throw NativeHandwritingError.invalidStyleData(style)
        }
        return value
    }

    private static func parse(
        _ data: Data,
        style: Int
    ) throws -> (header: String, payload: Data) {
        guard
            data.count >= 10,
            Array(data.prefix(6)) == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59]
        else {
            throw NativeHandwritingError.invalidStyleData(style)
        }

        let majorVersion = data[6]
        let headerStart: Int
        let headerLength: Int
        if majorVersion == 1 {
            headerStart = 10
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
        } else if majorVersion == 2 || majorVersion == 3, data.count >= 12 {
            headerStart = 12
            headerLength = Int(data[8])
                | (Int(data[9]) << 8)
                | (Int(data[10]) << 16)
                | (Int(data[11]) << 24)
        } else {
            throw NativeHandwritingError.invalidStyleData(style)
        }

        let payloadStart = headerStart + headerLength
        guard
            payloadStart <= data.count,
            let header = String(
                data: data[headerStart ..< payloadStart],
                encoding: .ascii
            ),
            header.contains("'fortran_order': False")
        else {
            throw NativeHandwritingError.invalidStyleData(style)
        }
        return (header, data[payloadStart...])
    }

    private static func float32(
        _ bytes: UnsafeRawBufferPointer,
        offset: Int
    ) -> Float {
        let raw = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        return Float(bitPattern: UInt32(littleEndian: raw))
    }
}

struct HandwritingPageLayout: Equatable {
    let dimensions: PageDimensions
    let fontSize: Double
    let margin: Double
    let lineHeight: Double
    let contentWidth: Double
    let maximumCharactersPerLine: Int
    let maximumLines: Int

    init(
        format: PageFormat,
        orientation: PageOrientation,
        fontSize: Double
    ) {
        let dimensions = format.dimensions(orientation: orientation)
        let resolvedFontSize = min(max(fontSize, 18), 72)
        let margin = max(24, min(54, min(dimensions.width, dimensions.height) * 0.075))
        let lineHeight = resolvedFontSize * 1.55
        let contentWidth = max(dimensions.width - (margin * 2), resolvedFontSize * 2)
        let contentHeight = max(dimensions.height - (margin * 2), lineHeight)

        self.dimensions = dimensions
        self.fontSize = resolvedFontSize
        self.margin = margin
        self.lineHeight = lineHeight
        self.contentWidth = contentWidth
        maximumCharactersPerLine = min(
            NativeHandwritingGenerator.maximumCharactersPerLine,
            max(4, Int(floor(contentWidth / (resolvedFontSize * 0.34))))
        )
        maximumLines = max(1, Int(floor(contentHeight / lineHeight)))
    }
}

enum HandwritingTextLayouter {
    static func wrap(
        _ text: String,
        maximumCharactersPerLine: Int
    ) -> [String] {
        let limit = max(1, maximumCharactersPerLine)
        let paragraphs = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ).map(String.init)
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

                let candidate = currentLine.isEmpty
                    ? word
                    : currentLine + " " + word
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

private extension NativeHandwritingGenerator {
    struct Coordinate {
        var x: Double
        var y: Double
        let penUp: Float
    }

    static func layout(
        sampledLines: [[StrokeOffset]?],
        alignment: TextAlignment,
        pageLayout: HandwritingPageLayout,
        unit: CanvasUnit
    ) -> RenderDocument {
        var paths = [RenderedPath]()

        for (lineIndex, offsets) in sampledLines.enumerated() {
            guard let offsets, !offsets.isEmpty else { continue }

            var x = 0.0
            var y = 0.0
            var coordinates = offsets.map { offset -> Coordinate in
                x += Double(offset.x)
                y += Double(offset.y)
                return Coordinate(x: x, y: y, penUp: offset.penUp)
            }
            coordinates = denoise(coordinates)
            coordinates = align(coordinates)
            coordinates = coordinates.map {
                Coordinate(x: $0.x, y: -$0.y, penUp: $0.penUp)
            }

            let minimumX = coordinates.map(\.x).min() ?? 0
            let maximumX = coordinates.map(\.x).max() ?? minimumX
            let minimumY = coordinates.map(\.y).min() ?? 0
            let maximumY = coordinates.map(\.y).max() ?? minimumY
            let rawWidth = max(maximumX - minimumX, 1)
            let rawHeight = max(maximumY - minimumY, 1)
            let desiredScale = pageLayout.fontSize / 24
            let scale = min(
                desiredScale,
                pageLayout.contentWidth / rawWidth
            )
            let lineWidth = rawWidth * scale
            let lineDrawingHeight = rawHeight * scale
            let lineTop = pageLayout.margin
                + (Double(lineIndex) * pageLayout.lineHeight)
            let verticalOffset = lineTop
                + max((pageLayout.lineHeight - lineDrawingHeight) / 2, 0)

            let horizontalOffset: Double
            if alignment == .center {
                horizontalOffset = (pageLayout.dimensions.width - lineWidth) / 2
            } else {
                horizontalOffset = pageLayout.margin
            }

            for index in coordinates.indices {
                coordinates[index].x = ((coordinates[index].x - minimumX) * scale)
                    + horizontalOffset
                coordinates[index].y = ((coordinates[index].y - minimumY) * scale)
                    + verticalOffset
            }

            var previousPenUp: Float = 1
            let points = coordinates.map { coordinate -> RenderPoint in
                defer { previousPenUp = coordinate.penUp }
                return RenderPoint(
                    x: coordinate.x,
                    y: coordinate.y,
                    move: previousPenUp == 1
                )
            }
            paths.append(
                RenderedPath(
                    strokeColor: "#111827",
                    lineWidth: max(
                        0.75,
                        (pageLayout.fontSize / 18) * (scale / desiredScale)
                    ),
                    points: points
                )
            )
        }

        return RenderDocument(
            width: pageLayout.dimensions.width,
            height: pageLayout.dimensions.height,
            backgroundColor: "#FFFFFF",
            paths: paths,
            unit: unit
        )
    }

    static func denoise(_ coordinates: [Coordinate]) -> [Coordinate] {
        var result = coordinates
        var start = 0
        for index in coordinates.indices where coordinates[index].penUp == 1 {
            smoothStroke(in: &result, source: coordinates, range: start ... index)
            start = index + 1
        }
        if start < coordinates.count {
            smoothStroke(
                in: &result,
                source: coordinates,
                range: start ... (coordinates.count - 1)
            )
        }
        return result
    }

    static func smoothStroke(
        in result: inout [Coordinate],
        source: [Coordinate],
        range: ClosedRange<Int>
    ) {
        let coefficients = [-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]
        for index in range {
            var smoothedX = 0.0
            var smoothedY = 0.0
            for (coefficientIndex, coefficient) in coefficients.enumerated() {
                let sourceIndex = min(
                    range.upperBound,
                    max(range.lowerBound, index + coefficientIndex - 3)
                )
                smoothedX += coefficient * source[sourceIndex].x
                smoothedY += coefficient * source[sourceIndex].y
            }
            result[index].x = smoothedX / 21
            result[index].y = smoothedY / 21
        }
    }

    static func align(_ coordinates: [Coordinate]) -> [Coordinate] {
        guard coordinates.count > 1 else { return coordinates }
        let count = Double(coordinates.count)
        let meanX = coordinates.map(\.x).reduce(0, +) / count
        let meanY = coordinates.map(\.y).reduce(0, +) / count
        let varianceX = coordinates.reduce(0) {
            $0 + (($1.x - meanX) * ($1.x - meanX))
        }
        let covariance = coordinates.reduce(0) {
            $0 + (($1.x - meanX) * ($1.y - meanY))
        }
        let slope = varianceX > .ulpOfOne ? covariance / varianceX : 0
        let intercept = meanY - slope * meanX
        let theta = atan(slope)
        let cosine = cos(theta)
        let sine = sin(theta)

        return coordinates.map {
            Coordinate(
                x: ($0.x * cosine) + ($0.y * sine) - intercept,
                y: (-$0.x * sine) + ($0.y * cosine) - intercept,
                penUp: $0.penUp
            )
        }
    }
}

enum NativeSVGRenderer {
    static func data(for document: RenderDocument) -> Data {
        let unit = document.unit?.rawValue ?? ""
        var svg = """
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 \(number(document.width)) \(number(document.height))" width="\(number(document.width))\(unit)" height="\(number(document.height))\(unit)">
        <rect width="100%" height="100%" fill="\(document.backgroundColor)"/>

        """
        for path in document.paths where !path.points.isEmpty {
            let commands = path.points.map { point in
                "\(point.move ? "M" : "L") \(number(point.x)) \(number(point.y))"
            }.joined(separator: " ")
            svg += "<path d=\"\(commands)\" fill=\"none\" stroke=\"\(path.strokeColor)\" stroke-width=\"\(number(path.lineWidth))\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n"
        }
        svg += "</svg>\n"
        return Data(svg.utf8)
    }

    private static func number(_ value: Double) -> String {
        String(format: "%.3f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
