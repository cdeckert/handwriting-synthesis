import AppKit
import CoreGraphics
import CoreText
import Foundation

struct RGBColor {
    let red: CGFloat
    let green: CGFloat
    let blue: CGFloat

    init(hex: String) throws {
        let value = hex.hasPrefix("#") ? String(hex.dropFirst()) : hex
        guard value.count == 6, let number = UInt32(value, radix: 16) else {
            throw HandwritingError.invalidColor(hex)
        }
        red = CGFloat((number >> 16) & 0xFF) / 255
        green = CGFloat((number >> 8) & 0xFF) / 255
        blue = CGFloat(number & 0xFF) / 255
    }

    var cgColor: CGColor {
        CGColor(colorSpace: CGColorSpaceCreateDeviceRGB(), components: [red, green, blue, 1])!
    }
}

public enum HandwritingRenderer {
    public static func write(
        _ document: GeneratedDocument,
        format: OutputFormat,
        to outputURL: URL,
        pngScale: Double = 2
    ) throws -> [URL] {
        guard !document.pages.isEmpty else {
            throw HandwritingError.outputCreationFailed(outputURL.path)
        }
        try FileManager.default.createDirectory(
            at: outputURL.deletingLastPathComponent(),
            withIntermediateDirectories: true
        )
        switch format {
        case .pdf:
            try writePDF(document.pages, to: outputURL)
            return [outputURL]
        case .svg:
            return try writePages(document.pages, baseURL: outputURL, extension: "svg") { page in
                svgData(for: page)
            }
        case .png:
            return try writePages(document.pages, baseURL: outputURL, extension: "png") { page in
                try pngData(for: page, scale: pngScale)
            }
        }
    }

    private static func writePDF(_ pages: [RenderDocument], to url: URL) throws {
        var mediaBox = CGRect(x: 0, y: 0, width: pages[0].width, height: pages[0].height)
        guard
            let consumer = CGDataConsumer(url: url as CFURL),
            let context = CGContext(consumer: consumer, mediaBox: &mediaBox, nil)
        else { throw HandwritingError.outputCreationFailed(url.path) }

        for page in pages {
            context.beginPDFPage(nil)
            draw(page, in: context)
            context.endPDFPage()
        }
        context.closePDF()
    }

    private static func writePages(
        _ pages: [RenderDocument],
        baseURL: URL,
        extension fileExtension: String,
        data: (RenderDocument) throws -> Data
    ) throws -> [URL] {
        let stem = baseURL.deletingPathExtension().lastPathComponent
        let directory = baseURL.deletingLastPathComponent()
        return try pages.enumerated().map { index, page in
            let suffix = pages.count == 1 ? "" : String(format: "-%03d", index + 1)
            let url = directory.appendingPathComponent(stem + suffix).appendingPathExtension(fileExtension)
            try data(page).write(to: url, options: .atomic)
            return url
        }
    }

    private static func pngData(for page: RenderDocument, scale: Double) throws -> Data {
        let resolvedScale = min(max(scale, 0.5), 4)
        let width = Int(ceil(page.width * resolvedScale))
        let height = Int(ceil(page.height * resolvedScale))
        guard
            let bitmap = NSBitmapImageRep(
                bitmapDataPlanes: nil,
                pixelsWide: width,
                pixelsHigh: height,
                bitsPerSample: 8,
                samplesPerPixel: 4,
                hasAlpha: true,
                isPlanar: false,
                colorSpaceName: .deviceRGB,
                bytesPerRow: 0,
                bitsPerPixel: 0
            ),
            let context = NSGraphicsContext(bitmapImageRep: bitmap)?.cgContext
        else { throw HandwritingError.outputCreationFailed("PNG bitmap") }

        context.scaleBy(x: resolvedScale, y: resolvedScale)
        draw(page, in: context)
        guard let result = bitmap.representation(using: .png, properties: [:]) else {
            throw HandwritingError.outputCreationFailed("PNG encoding")
        }
        return result
    }

    private static func draw(_ page: RenderDocument, in context: CGContext) {
        context.saveGState()
        context.translateBy(x: 0, y: page.height)
        context.scaleBy(x: 1, y: -1)
        if let background = try? RGBColor(hex: page.backgroundColor) {
            context.setFillColor(background.cgColor)
            context.fill(CGRect(x: 0, y: 0, width: page.width, height: page.height))
        }
        context.setLineCap(.round)
        context.setLineJoin(.round)

        for path in page.paths where !path.points.isEmpty {
            context.beginPath()
            for point in path.points {
                if point.move {
                    context.move(to: CGPoint(x: point.x, y: point.y))
                } else {
                    context.addLine(to: CGPoint(x: point.x, y: point.y))
                }
            }
            if let ink = try? RGBColor(hex: path.strokeColor) {
                context.setStrokeColor(ink.cgColor)
            }
            context.setLineWidth(path.lineWidth)
            context.strokePath()
        }
        for glyph in page.glyphs {
            draw(glyph, in: context)
        }
        context.restoreGState()
    }

    private static func draw(_ glyph: RenderedGlyph, in context: CGContext) {
        let font = CTFontCreateWithName(
            glyph.fontName as CFString,
            glyph.fontSize,
            nil
        )
        let color = (try? RGBColor(hex: glyph.color))?.cgColor
            ?? CGColor(
                colorSpace: CGColorSpaceCreateDeviceRGB(),
                components: [0.09, 0.15, 0.33, 1]
            )!
        let attributed = NSAttributedString(
            string: glyph.text,
            attributes: [
                NSAttributedString.Key(kCTFontAttributeName as String): font,
                NSAttributedString.Key(kCTForegroundColorAttributeName as String): color,
            ]
        )
        let line = CTLineCreateWithAttributedString(attributed)
        context.saveGState()
        context.translateBy(x: glyph.x, y: glyph.baselineY)
        context.rotate(by: glyph.rotation)
        context.scaleBy(x: 1, y: -1)
        context.textPosition = .zero
        CTLineDraw(line, context)
        context.restoreGState()
    }

    private static func svgData(for document: RenderDocument) -> Data {
        var svg = """
        <?xml version="1.0" encoding="UTF-8"?>
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 \(number(document.width)) \(number(document.height))" width="\(number(document.width))pt" height="\(number(document.height))pt">
        <rect width="100%" height="100%" fill="\(document.backgroundColor)"/>

        """
        for path in document.paths where !path.points.isEmpty {
            let commands = path.points.map { point in
                "\(point.move ? "M" : "L") \(number(point.x)) \(number(point.y))"
            }.joined(separator: " ")
            svg += "<path d=\"\(commands)\" fill=\"none\" stroke=\"\(path.strokeColor)\" stroke-width=\"\(number(path.lineWidth))\" stroke-linecap=\"round\" stroke-linejoin=\"round\"/>\n"
        }
        for glyph in document.glyphs {
            let degrees = glyph.rotation * 180 / .pi
            svg += "<text x=\"\(number(glyph.x))\" y=\"\(number(glyph.baselineY))\" font-family=\"\(xmlEscaped(glyph.fontName))\" font-size=\"\(number(glyph.fontSize))\" fill=\"\(glyph.color)\" transform=\"rotate(\(number(degrees)) \(number(glyph.x)) \(number(glyph.baselineY)))\">\(xmlEscaped(glyph.text))</text>\n"
        }
        svg += "</svg>\n"
        return Data(svg.utf8)
    }

    private static func xmlEscaped(_ value: String) -> String {
        value
            .replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
            .replacingOccurrences(of: "'", with: "&apos;")
    }

    private static func number(_ value: Double) -> String {
        String(format: "%.3f", locale: Locale(identifier: "en_US_POSIX"), value)
    }
}
