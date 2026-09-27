import SwiftUI

struct HandwritingCanvas: View {
    let document: RenderDocument

    var body: some View {
        Canvas { context, size in
            guard document.width > 0, document.height > 0 else { return }
            let scale = min(
                size.width / document.width,
                size.height / document.height
            )
            let horizontalOffset = (size.width - (document.width * scale)) / 2
            let verticalOffset = (size.height - (document.height * scale)) / 2

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
        .clipShape(RoundedRectangle(cornerRadius: 8, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 8, style: .continuous)
                .stroke(.quaternary)
        }
        .shadow(color: .black.opacity(0.08), radius: 8, y: 3)
        .accessibilityLabel("Generated handwriting preview")
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
