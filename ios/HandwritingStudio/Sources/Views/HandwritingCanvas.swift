import SwiftUI

struct HandwritingCanvas: View {
    let document: RenderDocument

    var body: some View {
        Canvas { context, size in
            let bounds = drawingBounds
            guard bounds.width > 0, bounds.height > 0 else { return }

            let inset = 14.0
            let availableWidth = max(size.width - (inset * 2), 1)
            let availableHeight = max(size.height - (inset * 2), 1)
            let scale = min(
                availableWidth / bounds.width,
                availableHeight / bounds.height
            )
            let horizontalOffset = ((size.width - (bounds.width * scale)) / 2) - (bounds.minX * scale)
            let verticalOffset = ((size.height - (bounds.height * scale)) / 2) - (bounds.minY * scale)

            context.translateBy(x: horizontalOffset, y: verticalOffset)
            context.scaleBy(x: scale, y: scale)

            for renderedPath in document.paths {
                var path = Path()
                for point in renderedPath.points {
                    let position = CGPoint(x: point.x, y: point.y)
                    if point.move {
                        path.move(to: position)
                    } else {
                        path.addLine(to: position)
                    }
                }

                context.stroke(
                    path,
                    with: .color(Color(hex: renderedPath.strokeColor)),
                    style: StrokeStyle(
                        lineWidth: renderedPath.lineWidth,
                        lineCap: .round,
                        lineJoin: .round
                    )
                )
            }
        }
        .background(Color(hex: document.backgroundColor))
        .clipShape(RoundedRectangle(cornerRadius: 16, style: .continuous))
        .accessibilityLabel("Generated handwriting preview")
    }

    private var drawingBounds: CGRect {
        let points = document.paths.flatMap(\.points)
        guard
            let minimumX = points.map(\.x).min(),
            let maximumX = points.map(\.x).max(),
            let minimumY = points.map(\.y).min(),
            let maximumY = points.map(\.y).max()
        else {
            return CGRect(x: 0, y: 0, width: document.width, height: document.height)
        }

        let modelPadding = 12.0
        return CGRect(
            x: minimumX - modelPadding,
            y: minimumY - modelPadding,
            width: max((maximumX - minimumX) + (modelPadding * 2), 1),
            height: max((maximumY - minimumY) + (modelPadding * 2), 1)
        )
    }
}

private extension Color {
    init(hex: String) {
        let normalized = hex.trimmingCharacters(in: CharacterSet.alphanumerics.inverted)
        var value: UInt64 = 0
        Scanner(string: normalized).scanHexInt64(&value)

        let red: Double
        let green: Double
        let blue: Double
        if normalized.count == 6 {
            red = Double((value >> 16) & 0xFF) / 255
            green = Double((value >> 8) & 0xFF) / 255
            blue = Double(value & 0xFF) / 255
        } else {
            red = 0
            green = 0
            blue = 0
        }

        self.init(red: red, green: green, blue: blue)
    }
}
