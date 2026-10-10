import CoreGraphics
import UIKit

enum DocumentExportRenderer {
    static func data(
        for document: RenderDocument,
        format: ExportFormat
    ) throws -> Data {
        switch format {
        case .svg:
            return NativeSVGRenderer.data(for: document)
        case .pdf:
            return pdfData(for: document)
        case .png:
            guard let data = image(for: document).pngData() else {
                throw DocumentExportError.encodingFailed(format)
            }
            return data
        case .jpeg:
            guard let data = image(for: document).jpegData(compressionQuality: 0.92) else {
                throw DocumentExportError.encodingFailed(format)
            }
            return data
        }
    }

    private static func pdfData(for document: RenderDocument) -> Data {
        let bounds = CGRect(
            x: 0,
            y: 0,
            width: max(document.width, 1),
            height: max(document.height, 1)
        )
        return UIGraphicsPDFRenderer(bounds: bounds).pdfData { renderer in
            renderer.beginPage()
            draw(document, in: renderer.cgContext, scale: 1)
        }
    }

    private static func image(for document: RenderDocument) -> UIImage {
        let longestEdge = max(max(document.width, document.height), 1)
        let scale = max(1, min(2, 2_400 / longestEdge))
        let size = CGSize(
            width: max(document.width * scale, 1),
            height: max(document.height * scale, 1)
        )
        let rendererFormat = UIGraphicsImageRendererFormat()
        rendererFormat.opaque = true
        rendererFormat.scale = 1
        return UIGraphicsImageRenderer(size: size, format: rendererFormat).image { renderer in
            draw(document, in: renderer.cgContext, scale: scale)
        }
    }

    private static func draw(
        _ document: RenderDocument,
        in context: CGContext,
        scale: Double
    ) {
        context.saveGState()
        defer { context.restoreGState() }

        context.scaleBy(x: scale, y: scale)
        context.setFillColor(color(from: document.backgroundColor))
        context.fill(
            CGRect(
                x: 0,
                y: 0,
                width: document.width,
                height: document.height
            )
        )

        context.setLineCap(.round)
        context.setLineJoin(.round)

        for renderedPath in document.paths where !renderedPath.points.isEmpty {
            let path = CGMutablePath()
            for point in renderedPath.points {
                let position = CGPoint(x: point.x, y: point.y)
                if point.move {
                    path.move(to: position)
                } else {
                    path.addLine(to: position)
                }
            }
            context.addPath(path)
            context.setStrokeColor(color(from: renderedPath.strokeColor))
            context.setLineWidth(renderedPath.lineWidth)
            context.strokePath()
        }
    }

    private static func color(from hex: String) -> CGColor {
        let normalized = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: normalized).scanHexInt64(&value)
        guard normalized.count == 6 else {
            return UIColor.black.cgColor
        }
        return UIColor(
            red: CGFloat((value >> 16) & 0xFF) / 255,
            green: CGFloat((value >> 8) & 0xFF) / 255,
            blue: CGFloat(value & 0xFF) / 255,
            alpha: 1
        ).cgColor
    }
}

private enum DocumentExportError: LocalizedError {
    case encodingFailed(ExportFormat)

    var errorDescription: String? {
        switch self {
        case let .encodingFailed(format):
            "The \(format.title) file could not be created."
        }
    }
}
