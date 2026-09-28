import CoreText
import Foundation

enum UnicodeHandwritingGenerator {
    private struct GlyphSpec {
        let text: String
        let fontName: String
        let advance: Double
        let isWhitespace: Bool
    }

    private struct LineSpec {
        var glyphs: [GlyphSpec]
        var width: Double
    }

    static func render(
        text: String,
        options: GenerationOptions
    ) throws -> [RenderDocument] {
        let layout = HandwritingPageLayout(
            format: options.pageFormat,
            orientation: options.pageOrientation,
            fontSize: options.fontSize
        )
        _ = try RGBColor(hex: options.inkColor)
        _ = try RGBColor(hex: options.backgroundColor)

        let baseFont = CTFontCreateWithName(
            options.unicodeFont as CFString,
            layout.fontSize,
            nil
        )
        let lines = wrap(
            text,
            baseFont: baseFont,
            fontSize: layout.fontSize,
            maximumWidth: layout.contentWidth
        )
        var random = UnicodeRandomSource(
            seed: options.seed ?? UInt64.random(in: .min ... .max)
        )
        var pages = [RenderDocument]()

        for pageStart in stride(from: 0, to: lines.count, by: layout.maximumLines) {
            let pageEnd = min(pageStart + layout.maximumLines, lines.count)
            let pageLines = lines[pageStart ..< pageEnd]
            var glyphs = [RenderedGlyph]()

            for (localLineIndex, line) in pageLines.enumerated() {
                var cursorX = options.alignment == .center
                    ? (layout.dimensions.width - line.width) / 2
                    : layout.margin
                let baseline = layout.margin
                    + (Double(localLineIndex) * layout.lineHeight)
                    + layout.fontSize

                for glyph in line.glyphs {
                    if !glyph.isWhitespace {
                        let sizeVariation = 0.985 + (random.unitInterval() * 0.03)
                        let xJitter = (random.unitInterval() - 0.5) * 0.8
                        let yJitter = (random.unitInterval() - 0.5) * 2.2
                        let rotation = (random.unitInterval() - 0.5) * 0.035
                        glyphs.append(
                            RenderedGlyph(
                                text: glyph.text,
                                x: cursorX + xJitter,
                                baselineY: baseline + yJitter,
                                fontName: glyph.fontName,
                                fontSize: layout.fontSize * sizeVariation,
                                rotation: rotation,
                                color: options.inkColor
                            )
                        )
                    }
                    cursorX += glyph.advance
                }
            }

            pages.append(
                RenderDocument(
                    width: layout.dimensions.width,
                    height: layout.dimensions.height,
                    backgroundColor: options.backgroundColor,
                    glyphs: glyphs
                )
            )
        }
        return pages
    }

    private static func wrap(
        _ text: String,
        baseFont: CTFont,
        fontSize: Double,
        maximumWidth: Double
    ) -> [LineSpec] {
        let paragraphs = text.split(
            separator: "\n",
            omittingEmptySubsequences: false
        ).map(String.init)
        var result = [LineSpec]()
        let space = makeGlyph(" ", baseFont: baseFont, fontSize: fontSize)

        for paragraph in paragraphs {
            let words = paragraph.split(whereSeparator: { $0.isWhitespace }).map(String.init)
            guard !words.isEmpty else {
                result.append(LineSpec(glyphs: [], width: 0))
                continue
            }

            var current = LineSpec(glyphs: [], width: 0)
            for word in words {
                let wordGlyphs = word.map {
                    makeGlyph(String($0), baseFont: baseFont, fontSize: fontSize)
                }
                let wordWidth = wordGlyphs.reduce(0) { $0 + $1.advance }
                let separatorWidth = current.glyphs.isEmpty ? 0 : space.advance

                if current.width + separatorWidth + wordWidth <= maximumWidth {
                    if !current.glyphs.isEmpty {
                        current.glyphs.append(space)
                        current.width += space.advance
                    }
                    current.glyphs.append(contentsOf: wordGlyphs)
                    current.width += wordWidth
                    continue
                }

                if !current.glyphs.isEmpty {
                    result.append(current)
                    current = LineSpec(glyphs: [], width: 0)
                }

                if wordWidth <= maximumWidth {
                    current = LineSpec(glyphs: wordGlyphs, width: wordWidth)
                } else {
                    for glyph in wordGlyphs {
                        if current.width + glyph.advance > maximumWidth, !current.glyphs.isEmpty {
                            result.append(current)
                            current = LineSpec(glyphs: [], width: 0)
                        }
                        current.glyphs.append(glyph)
                        current.width += glyph.advance
                    }
                }
            }
            if !current.glyphs.isEmpty {
                result.append(current)
            }
        }
        return result
    }

    private static func makeGlyph(
        _ text: String,
        baseFont: CTFont,
        fontSize: Double
    ) -> GlyphSpec {
        let string = text as CFString
        let range = CFRange(location: 0, length: CFStringGetLength(string))
        let resolvedFont = CTFontCreateForString(baseFont, string, range)
        let attributes: [NSAttributedString.Key: Any] = [
            NSAttributedString.Key(kCTFontAttributeName as String): resolvedFont,
            NSAttributedString.Key(kCTKernAttributeName as String): fontSize * 0.012,
        ]
        let line = CTLineCreateWithAttributedString(
            NSAttributedString(string: text, attributes: attributes)
        )
        let width = max(CTLineGetTypographicBounds(line, nil, nil, nil), fontSize * 0.14)
        return GlyphSpec(
            text: text,
            fontName: CTFontCopyPostScriptName(resolvedFont) as String,
            advance: width,
            isWhitespace: text.allSatisfy(\.isWhitespace)
        )
    }
}

private struct UnicodeRandomSource {
    private var state: UInt64

    init(seed: UInt64) {
        state = seed &+ 0x9E3779B97F4A7C15
    }

    mutating func unitInterval() -> Double {
        state &+= 0x9E3779B97F4A7C15
        var value = state
        value = (value ^ (value >> 30)) &* 0xBF58476D1CE4E5B9
        value = (value ^ (value >> 27)) &* 0x94D049BB133111EB
        value ^= value >> 31
        return Double(value >> 11) / Double(1 << 53)
    }
}
