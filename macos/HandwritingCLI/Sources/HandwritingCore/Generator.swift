import CoreML
import Foundation

public final class HandwritingGenerator {
    public static let maximumTextLength = 50_000
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

    public init() throws {
        guard let modelURL = Self.resourceURL(
            name: "HandwritingStep",
            extension: "mlmodelc",
            subdirectory: "Models"
        ) else {
            throw HandwritingError.outputCreationFailed("bundled Core ML model")
        }
        let configuration = MLModelConfiguration()
        // Repeated one-step predictions can exhaust E5/Metal IOSurface buffers
        // on long documents. CPU execution remains fully local and is stable.
        configuration.computeUnits = .cpuOnly
        model = try HandwritingStep(contentsOf: modelURL, configuration: configuration)
    }

    public func generate(
        text: String,
        options: GenerationOptions = GenerationOptions(),
        progress: @escaping (Double) -> Void = { _ in }
    ) throws -> GeneratedDocument {
        let normalized = try Self.normalize(text)
        guard normalized.count <= Self.maximumTextLength else {
            throw HandwritingError.textTooLong(Self.maximumTextLength)
        }
        let resolvedEngine: GenerationEngine = switch options.engine {
        case .auto:
            Self.canRenderNeurally(normalized) ? .neural : .unicode
        case .neural:
            .neural
        case .unicode:
            .unicode
        }
        if resolvedEngine == .unicode {
            let pages = try UnicodeHandwritingGenerator.render(
                text: normalized,
                options: options
            )
            progress(1)
            return GeneratedDocument(
                pages: pages,
                normalizedText: normalized,
                engine: .unicode
            )
        }
        for character in normalized where character != "\n" {
            guard Self.characterNumbers[character] != nil else {
                throw HandwritingError.unsupportedCharacter(character)
            }
        }
        guard HandwritingStyle.bundled.indices.contains(options.style) else {
            throw HandwritingError.missingStyle(options.style)
        }
        _ = try RGBColor(hex: options.inkColor)
        _ = try RGBColor(hex: options.backgroundColor)

        let pageLayout = HandwritingPageLayout(
            format: options.pageFormat,
            orientation: options.pageOrientation,
            fontSize: options.fontSize
        )
        let lines = HandwritingTextLayouter.wrap(
            normalized,
            maximumCharactersPerLine: pageLayout.maximumCharactersPerLine
        )
        let style = try styleStore.load(id: options.style)
        let nonemptyLines = lines.filter { !$0.isEmpty }
        let totalSteps = nonemptyLines.reduce(0) { total, line in
            total + style.strokes.count + max(line.count * 40, 40)
        }
        var reporter = GenerationProgressReporter(total: max(totalSteps, 1), callback: progress)
        reporter.start()
        var random = SeededRandomSource(seed: options.seed ?? UInt64.random(in: .min ... .max))
        var sampledLines = [[StrokeOffset]?]()
        sampledLines.reserveCapacity(lines.count)

        for line in lines {
            if line.isEmpty {
                sampledLines.append(nil)
            } else {
                sampledLines.append(
                    try sample(
                        line: line,
                        style: style,
                        bias: Float(min(max(options.bias, 0.1), 1.5)),
                        random: &random,
                        progress: &reporter
                    )
                )
            }
        }
        reporter.finish()

        let pages = stride(from: 0, to: sampledLines.count, by: pageLayout.maximumLines).map { start in
            let end = min(start + pageLayout.maximumLines, sampledLines.count)
            return Self.layout(
                sampledLines: Array(sampledLines[start ..< end]),
                alignment: options.alignment,
                pageLayout: pageLayout,
                inkColor: options.inkColor,
                backgroundColor: options.backgroundColor
            )
        }
        return GeneratedDocument(
            pages: pages,
            normalizedText: normalized,
            engine: .neural
        )
    }

    public static func normalize(_ text: String) throws -> String {
        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            throw HandwritingError.emptyText
        }

        return text
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")
            .replacingOccurrences(of: "\t", with: "    ")
            .precomposedStringWithCanonicalMapping
    }

    public static func canRenderNeurally(_ text: String) -> Bool {
        text.allSatisfy { $0 == "\n" || Self.characterNumbers[$0] != nil }
    }

    private func sample(
        line: String,
        style: HandwritingStyleSample,
        bias: Float,
        random: inout SeededRandomSource,
        progress: inout GenerationProgressReporter
    ) throws -> [StrokeOffset] {
        let encoded = try encode(style.characters + " " + line)
        var state = RecurrentState()
        var lastParameters: [Float]?

        for stroke in style.strokes {
            let output = try step(stroke: stroke, encoded: encoded, state: state)
            state = output.state
            lastParameters = output.gmmParameters
            progress.advance()
        }
        guard let primedParameters = lastParameters else {
            throw HandwritingError.invalidModelOutput
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
            let output = try step(stroke: previousStroke, encoded: encoded, state: state)
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
            throw HandwritingError.invalidModelOutput
        }
        return result
    }

    private func encode(_ text: String) throws -> EncodedText {
        var values = [Int32]()
        values.reserveCapacity(text.count + 1)
        for character in text {
            guard let value = Self.characterNumbers[character] else {
                throw HandwritingError.unsupportedCharacter(character)
            }
            values.append(value)
        }
        values.append(0)
        guard values.count <= 120 else {
            throw HandwritingError.invalidModelOutput
        }
        let length = values.count
        values.append(contentsOf: repeatElement(0, count: 120 - values.count))
        return EncodedText(values: values, length: length)
    }

    private func step(stroke: StrokeOffset, encoded: EncodedText, state: RecurrentState) throws -> StepResult {
        try autoreleasepool {
            let output = try model.prediction(
                stroke: MLShapedArray(scalars: [stroke.x, stroke.y, stroke.penUp], shape: [1, 3]),
                chars: MLShapedArray(scalars: encoded.values, shape: [1, 120]),
                chars_len: MLShapedArray(scalars: [Int32(encoded.length)], shape: [1]),
                h1: state.shaped(state.h1), c1: state.shaped(state.c1),
                h2: state.shaped(state.h2), c2: state.shaped(state.c2),
                h3: state.shaped(state.h3), c3: state.shaped(state.c3),
                kappa: MLShapedArray(scalars: state.kappa, shape: [1, 10]),
                window: MLShapedArray(scalars: state.window, shape: [1, 73])
            )
            let parameters = Array(output.gmm_paramsShapedArray.scalars)
            guard parameters.count == 121, parameters.allSatisfy(\.isFinite) else {
                throw HandwritingError.invalidModelOutput
            }
            return StepResult(gmmParameters: parameters, state: RecurrentState(output: output))
        }
    }

    private static func sampleStroke(
        parameters: [Float],
        bias: Float,
        random: inout SeededRandomSource
    ) throws -> StrokeOffset {
        guard parameters.count == 121 else { throw HandwritingError.invalidModelOutput }
        let scaledLogits = parameters[0 ..< 20].map { Double($0 * (1 + bias)) }
        let largestLogit = scaledLogits.max() ?? 0
        var weights = scaledLogits.map { exp($0 - largestLogit) }
        let total = weights.reduce(0, +)
        guard total.isFinite, total > 0 else { throw HandwritingError.invalidModelOutput }
        weights = weights.map {
            let probability = $0 / total
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
        let y = meanY + sigmaY * (rho * normals.0 + sqrt(max(0, 1 - (rho * rho))) * normals.1)
        let rawEndProbability = sigmoid(Double(parameters[120]))
        let endProbability = rawEndProbability < 0.01 ? 0 : rawEndProbability
        guard x.isFinite, y.isFinite else { throw HandwritingError.invalidModelOutput }
        return StrokeOffset(
            x: Float(x),
            y: Float(y),
            penUp: random.unitInterval() < endProbability ? 1 : 0
        )
    }

    private static func sigmoid(_ value: Double) -> Double {
        if value >= 0 { return 1 / (1 + exp(-value)) }
        let exponential = exp(value)
        return exponential / (1 + exponential)
    }

    static func resourceURL(name: String, extension ext: String, subdirectory: String) -> URL? {
        Bundle.module.url(forResource: name, withExtension: ext, subdirectory: subdirectory)
            ?? Bundle.module.url(forResource: name, withExtension: ext)
    }
}

private struct EncodedText { let values: [Int32]; let length: Int }
private struct StepResult { let gmmParameters: [Float]; let state: RecurrentState }

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

private struct StrokeOffset { let x: Float; let y: Float; let penUp: Float }

private struct SeededRandomSource {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed &+ 0x9E3779B97F4A7C15
    }

    mutating func next() -> UInt64 {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        return value ^ (value >> 31)
    }

    mutating func unitInterval() -> Double {
        Double(next() >> 11) / Double(1 << 53)
    }

    mutating func weightedIndex(_ weights: [Double]) -> Int {
        let total = weights.reduce(0, +)
        guard total > 0 else { return weights.indices.max(by: { weights[$0] < weights[$1] }) ?? 0 }
        var cursor = unitInterval() * total
        for (index, weight) in weights.enumerated() {
            cursor -= weight
            if cursor <= 0 { return index }
        }
        return weights.indices.last ?? 0
    }

    mutating func normalPair() -> (Double, Double) {
        let first = max(unitInterval(), Double.leastNonzeroMagnitude)
        let second = unitInterval()
        let magnitude = sqrt(-2 * log(first))
        return (magnitude * cos(2 * .pi * second), magnitude * sin(2 * .pi * second))
    }
}

private struct GenerationProgressReporter {
    let total: Int
    let callback: (Double) -> Void
    private var completed = 0
    private var lastReported = -1.0

    init(total: Int, callback: @escaping (Double) -> Void) {
        self.total = total
        self.callback = callback
    }

    mutating func start() { callback(0); lastReported = 0 }
    mutating func advance() {
        completed += 1
        let value = min(Double(completed) / Double(total), 0.99)
        if value - lastReported >= 0.01 { callback(value); lastReported = value }
    }
    func finish() { callback(1) }
}

private struct HandwritingStyleSample { let characters: String; let strokes: [StrokeOffset] }

private final class HandwritingStyleStore {
    private var cache = [Int: HandwritingStyleSample]()

    func load(id: Int) throws -> HandwritingStyleSample {
        if let cached = cache[id] { return cached }
        guard (0 ... 12).contains(id) else { throw HandwritingError.missingStyle(id) }
        guard
            let strokeURL = HandwritingGenerator.resourceURL(name: "style-\(id)-strokes", extension: "npy", subdirectory: "Styles"),
            let characterURL = HandwritingGenerator.resourceURL(name: "style-\(id)-chars", extension: "npy", subdirectory: "Styles")
        else { throw HandwritingError.missingStyle(id) }

        let strokes = try Self.decodeStrokes(Data(contentsOf: strokeURL), style: id)
        let characters = try Self.decodeCharacters(Data(contentsOf: characterURL), style: id)
        let sample = HandwritingStyleSample(characters: characters, strokes: strokes)
        cache[id] = sample
        return sample
    }

    private static func decodeStrokes(_ data: Data, style: Int) throws -> [StrokeOffset] {
        let parsed = try parse(data, style: style)
        guard parsed.header.contains("'<f4'"), parsed.payload.count.isMultiple(of: 12) else {
            throw HandwritingError.invalidStyleData(style)
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

    private static func decodeCharacters(_ data: Data, style: Int) throws -> String {
        let parsed = try parse(data, style: style)
        guard parsed.header.contains("'|S") else { throw HandwritingError.invalidStyleData(style) }
        let bytes = parsed.payload.prefix { $0 != 0 }
        guard let value = String(data: bytes, encoding: .utf8) else {
            throw HandwritingError.invalidStyleData(style)
        }
        return value
    }

    private static func parse(_ data: Data, style: Int) throws -> (header: String, payload: Data) {
        guard data.count >= 10, Array(data.prefix(6)) == [0x93, 0x4E, 0x55, 0x4D, 0x50, 0x59] else {
            throw HandwritingError.invalidStyleData(style)
        }
        let majorVersion = data[6]
        let headerStart: Int
        let headerLength: Int
        if majorVersion == 1 {
            headerStart = 10
            headerLength = Int(data[8]) | (Int(data[9]) << 8)
        } else if majorVersion == 2 || majorVersion == 3, data.count >= 12 {
            headerStart = 12
            headerLength = Int(data[8]) | (Int(data[9]) << 8) | (Int(data[10]) << 16) | (Int(data[11]) << 24)
        } else {
            throw HandwritingError.invalidStyleData(style)
        }
        let payloadStart = headerStart + headerLength
        guard
            payloadStart <= data.count,
            let header = String(data: data[headerStart ..< payloadStart], encoding: .ascii),
            header.contains("'fortran_order': False")
        else { throw HandwritingError.invalidStyleData(style) }
        return (header, data[payloadStart...])
    }

    private static func float32(_ bytes: UnsafeRawBufferPointer, offset: Int) -> Float {
        let raw = bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        return Float(bitPattern: UInt32(littleEndian: raw))
    }
}

private extension HandwritingGenerator {
    struct Coordinate { var x: Double; var y: Double; let penUp: Float }

    static func layout(
        sampledLines: [[StrokeOffset]?],
        alignment: TextAlignment,
        pageLayout: HandwritingPageLayout,
        inkColor: String,
        backgroundColor: String
    ) -> RenderDocument {
        var paths = [RenderedPath]()
        for (lineIndex, offsets) in sampledLines.enumerated() {
            guard let offsets, !offsets.isEmpty else { continue }
            var x = 0.0
            var y = 0.0
            var coordinates = offsets.map { offset -> Coordinate in
                x += Double(offset.x); y += Double(offset.y)
                return Coordinate(x: x, y: y, penUp: offset.penUp)
            }
            coordinates = denoise(coordinates)
            coordinates = align(coordinates).map { Coordinate(x: $0.x, y: -$0.y, penUp: $0.penUp) }

            let minimumX = coordinates.map(\.x).min() ?? 0
            let maximumX = coordinates.map(\.x).max() ?? minimumX
            let minimumY = coordinates.map(\.y).min() ?? 0
            let maximumY = coordinates.map(\.y).max() ?? minimumY
            let rawWidth = max(maximumX - minimumX, 1)
            let rawHeight = max(maximumY - minimumY, 1)
            let desiredScale = pageLayout.fontSize / 24
            let scale = min(desiredScale, pageLayout.contentWidth / rawWidth)
            let lineWidth = rawWidth * scale
            let lineDrawingHeight = rawHeight * scale
            let lineTop = pageLayout.margin + (Double(lineIndex) * pageLayout.lineHeight)
            let verticalOffset = lineTop + max((pageLayout.lineHeight - lineDrawingHeight) / 2, 0)
            let horizontalOffset = alignment == .center
                ? (pageLayout.dimensions.width - lineWidth) / 2
                : pageLayout.margin

            for index in coordinates.indices {
                coordinates[index].x = ((coordinates[index].x - minimumX) * scale) + horizontalOffset
                coordinates[index].y = ((coordinates[index].y - minimumY) * scale) + verticalOffset
            }
            var previousPenUp: Float = 1
            let points = coordinates.map { coordinate -> RenderPoint in
                defer { previousPenUp = coordinate.penUp }
                return RenderPoint(x: coordinate.x, y: coordinate.y, move: previousPenUp == 1)
            }
            paths.append(
                RenderedPath(
                    strokeColor: inkColor,
                    lineWidth: max(0.75, (pageLayout.fontSize / 18) * (scale / desiredScale)),
                    points: points
                )
            )
        }
        return RenderDocument(
            width: pageLayout.dimensions.width,
            height: pageLayout.dimensions.height,
            backgroundColor: backgroundColor,
            paths: paths
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
            smoothStroke(in: &result, source: coordinates, range: start ... (coordinates.count - 1))
        }
        return result
    }

    static func smoothStroke(in result: inout [Coordinate], source: [Coordinate], range: ClosedRange<Int>) {
        let coefficients = [-2.0, 3.0, 6.0, 7.0, 6.0, 3.0, -2.0]
        for index in range {
            var smoothedX = 0.0
            var smoothedY = 0.0
            for (coefficientIndex, coefficient) in coefficients.enumerated() {
                let sourceIndex = min(range.upperBound, max(range.lowerBound, index + coefficientIndex - 3))
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
        let varianceX = coordinates.reduce(0) { $0 + (($1.x - meanX) * ($1.x - meanX)) }
        let covariance = coordinates.reduce(0) { $0 + (($1.x - meanX) * ($1.y - meanY)) }
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
