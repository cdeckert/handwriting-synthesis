import CoreML
import XCTest
@testable import HandwritingStudio

final class HandwritingStudioTests: XCTestCase {
    func testBundledStylesAreAvailable() {
        XCTAssertEqual(HandwritingStyle.bundled.count, 13)
        XCTAssertEqual(HandwritingStyle.bundled.first?.label, "Light Loops")
        XCTAssertEqual(HandwritingStyle.bundled[9].label, "Bold Slant")
        XCTAssertEqual(HandwritingStyle.bundled.last?.id, 12)
    }

    func testNativeVectorDocumentDecoding() throws {
        let json = ##"{"width":1000,"height":120,"backgroundColor":"#FFFFFF","paths":[{"strokeColor":"black","lineWidth":2,"points":[{"x":12.5,"y":34,"move":true}]}]}"##
        let document = try JSONDecoder().decode(RenderDocument.self, from: Data(json.utf8))

        XCTAssertEqual(document.width, 1000)
        XCTAssertEqual(document.paths.first?.points.first, RenderPoint(x: 12.5, y: 34, move: true))
    }

    func testPageFormatsAndOrientation() {
        let portrait = PageFormat.a4.dimensions(orientation: .portrait)
        let landscape = PageFormat.a4.dimensions(orientation: .landscape)

        XCTAssertEqual(portrait.width, 595.28, accuracy: 0.01)
        XCTAssertEqual(portrait.height, 841.89, accuracy: 0.01)
        XCTAssertEqual(landscape.width, portrait.height, accuracy: 0.01)
        XCTAssertEqual(landscape.height, portrait.width, accuracy: 0.01)
        XCTAssertEqual(PageFormat.desktopHD.defaultOrientation, .landscape)
    }

    func testTextWrapsAtWordsAndKeepsExplicitBreaks() {
        let lines = HandwritingTextLayouter.wrap(
            "The quick brown fox\nNext paragraph",
            maximumCharactersPerLine: 11
        )

        XCTAssertEqual(lines, ["The quick", "brown fox", "Next", "paragraph"])
        XCTAssertTrue(lines.allSatisfy { $0.count <= 11 })
    }

    func testLongWordsWrapWithoutExceedingLineLimit() {
        let lines = HandwritingTextLayouter.wrap(
            "abcdefghijk",
            maximumCharactersPerLine: 4
        )

        XCTAssertEqual(lines, ["abcd", "efgh", "ijk"])
    }

    func testGermanAndFrenchCharactersPrepareForTheFixedModelAlphabet() throws {
        let prepared = try HandwritingOrthography.prepare("Füße à Noël, cœur")

        XCTAssertEqual(prepared.modelText, "Fuse a Noel, coeur")
        XCTAssertEqual(
            prepared.marks.map(\.diacritic),
            [.diaeresis, .sharpS, .grave, .diaeresis]
        )
        XCTAssertEqual(prepared.marks.map(\.characterIndex), [1, 2, 5, 9])
    }

    func testSupportedGermanAndFrenchCharacterInventory() throws {
        XCTAssertNoThrow(
            try HandwritingOrthography.prepare(
                "ÄÖÜäöüß ÀÂÇÉÈÊËÎÏÔÙÛÜŸ àâçéèêëîïôùûüÿ ŒœÆæ"
            )
        )
    }

    func testBundledCoreMLStepRuns() throws {
        let model = try HandwritingStep(configuration: MLModelConfiguration())
        let output = try model.prediction(
            stroke: MLShapedArray<Float>(repeating: 0, shape: [1, 3]),
            chars: MLShapedArray<Int32>(repeating: 0, shape: [1, 120]),
            chars_len: MLShapedArray<Int32>(scalars: [1], shape: [1]),
            h1: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            c1: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            h2: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            c2: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            h3: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            c3: MLShapedArray<Float>(repeating: 0, shape: [1, 400]),
            kappa: MLShapedArray<Float>(repeating: 0, shape: [1, 10]),
            window: MLShapedArray<Float>(repeating: 0, shape: [1, 73])
        )

        XCTAssertEqual(output.gmm_params.count, 121)
        XCTAssertTrue(output.gmm_paramsShapedArray.scalars.allSatisfy(\.isFinite))
        XCTAssertEqual(output.next_phi.count, 120)
    }

    func testOfflineGeneratorProducesHandwriting() throws {
        let generator = try NativeHandwritingGenerator()
        let progress = ProgressRecorder()
        let document = try generator.render(
            request: GenerationRequest(
                text: "Hi",
                style: 0,
                alignment: .center,
                pageFormat: .a4,
                pageOrientation: .portrait,
                fontSize: 36
            ),
            bias: 0.75,
            progress: { progress.append($0) }
        )

        let points = try XCTUnwrap(document.paths.first?.points)
        XCTAssertGreaterThan(points.count, 10)
        XCTAssertTrue(points.allSatisfy { $0.x.isFinite && $0.y.isFinite })
        XCTAssertTrue(points.contains(where: \.move))

        let svg = String(decoding: NativeSVGRenderer.data(for: document), as: UTF8.self)
        XCTAssertTrue(svg.contains("<svg"))
        XCTAssertTrue(svg.contains("<path"))
        XCTAssertTrue(svg.contains("width=\"595.280pt\""))
        XCTAssertEqual(progress.values.first, 0)
        XCTAssertEqual(progress.values.last, 1)
        XCTAssertGreaterThan(progress.values.count, 2)
    }

    func testOfflineGeneratorRendersUnicodeMarksAsVectorPaths() throws {
        let generator = try NativeHandwritingGenerator()
        let document = try generator.render(
            request: GenerationRequest(
                text: "Füße café",
                style: 0,
                alignment: .left,
                pageFormat: .a4,
                pageOrientation: .portrait,
                fontSize: 36
            ),
            bias: 0.75
        )

        XCTAssertGreaterThanOrEqual(document.paths.count, 4)
        XCTAssertTrue(document.paths.dropFirst().allSatisfy { !$0.points.isEmpty })
    }
}

private final class ProgressRecorder: @unchecked Sendable {
    private let lock = NSLock()
    private var recordedValues = [Double]()

    var values: [Double] {
        lock.withLock { recordedValues }
    }

    func append(_ value: Double) {
        lock.withLock {
            recordedValues.append(value)
        }
    }
}
