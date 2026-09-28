import Foundation
import Testing
@testable import HandwritingCore

@Test func wrapsTextAndKeepsParagraphBreaks() {
    let lines = HandwritingTextLayouter.wrap(
        "The quick brown fox\nNext paragraph",
        maximumCharactersPerLine: 11
    )
    #expect(lines == ["The quick", "brown fox", "Next", "paragraph"])
}

@Test func preservesGermanTextExactly() throws {
    let result = try HandwritingGenerator.normalize("Übermäßig große Züge – schön!")
    #expect(result == "Übermäßig große Züge – schön!")
}

@Test func detectsWhenUnicodeRenderingIsRequired() {
    #expect(HandwritingGenerator.canRenderNeurally("Hello world"))
    #expect(!HandwritingGenerator.canRenderNeurally("Grüße für 5 €"))
}

@Test func pageDimensionsAreCorrect() {
    let portrait = PageFormat.a4.dimensions(orientation: .portrait)
    let landscape = PageFormat.a4.dimensions(orientation: .landscape)
    #expect(abs(portrait.width - 595.28) < 0.01)
    #expect(landscape.width == portrait.height)
    #expect(landscape.height == portrait.width)
}

@Test func bundledModelProducesDeterministicOutput() throws {
    let generator = try HandwritingGenerator()
    let options = GenerationOptions(style: 0, fontSize: 36, seed: 42)
    let first = try generator.generate(text: "Hi", options: options)
    let second = try generator.generate(text: "Hi", options: options)
    #expect(first.pages.count == 1)
    #expect(first.engine == .neural)
    #expect(first == second)
    #expect((first.pages.first?.paths.first?.points.count ?? 0) > 10)
}

@Test func unicodeEnginePreservesSpecialCharacters() throws {
    let generator = try HandwritingGenerator()
    let options = GenerationOptions(engine: .auto, fontSize: 36, seed: 42)
    let result = try generator.generate(
        text: "Ärger, Öl, Grüße, 25 € – déjà vu!",
        options: options
    )
    #expect(result.engine == .unicode)
    #expect(result.normalizedText == "Ärger, Öl, Grüße, 25 € – déjà vu!")
    let renderedText = result.pages.flatMap(\.glyphs).map(\.text).joined()
    for character in ["Ä", "Ö", "ü", "ß", "€", "–", "é"] {
        #expect(renderedText.contains(character))
    }
}
