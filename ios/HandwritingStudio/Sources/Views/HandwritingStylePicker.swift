import SwiftUI

struct HandwritingStylePicker: View {
    let styles: [HandwritingStyle]
    @Binding var selection: Int?

    var body: some View {
        ScrollView(.horizontal) {
            LazyHStack(spacing: 10) {
                ForEach(styles) { style in
                    Button {
                        selection = style.id
                    } label: {
                        VStack(alignment: .leading, spacing: 6) {
                            HandwritingStyleSampleView(styleID: style.id)
                                .frame(width: 106, height: 42)
                                .padding(.horizontal, 6)
                                .background(Color(uiColor: .systemBackground))
                                .clipShape(RoundedRectangle(cornerRadius: 9, style: .continuous))

                            HStack(spacing: 4) {
                                Text(style.label)
                                    .font(.caption.weight(.semibold))
                                    .lineLimit(1)
                                if style.isCustom {
                                    Image(systemName: "person.crop.circle.fill")
                                        .font(.caption2)
                                }
                            }

                            Text(style.detail)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                        .frame(width: 118, alignment: .leading)
                        .padding(9)
                        .background(
                            selection == style.id
                                ? Color.accentColor.opacity(0.12)
                                : Color(uiColor: .secondarySystemGroupedBackground)
                        )
                        .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 14, style: .continuous)
                                .stroke(
                                    selection == style.id ? Color.accentColor : Color.clear,
                                    lineWidth: 1.5
                                )
                        }
                    }
                    .buttonStyle(.plain)
                    .accessibilityLabel("\(style.label), \(style.detail)")
                    .accessibilityAddTraits(selection == style.id ? .isSelected : [])
                }
            }
            .padding(.horizontal, 1)
        }
        .scrollIndicators(.hidden)
        .accessibilityLabel("Writing style")
    }
}

private struct HandwritingStyleSampleView: View {
    let styleID: Int

    private var paths: [[CGPoint]] {
        HandwritingStylePreviewRepository.paths(for: styleID)
    }

    var body: some View {
        Canvas { context, size in
            let points = paths.flatMap { $0 }
            guard
                let minimumX = points.map(\.x).min(),
                let maximumX = points.map(\.x).max(),
                let minimumY = points.map(\.y).min(),
                let maximumY = points.map(\.y).max()
            else { return }

            let width = max(maximumX - minimumX, 1)
            let height = max(maximumY - minimumY, 1)
            let scale = min((size.width - 4) / width, (size.height - 4) / height)
            let offsetX = ((size.width - width * scale) / 2) - minimumX * scale
            let offsetY = ((size.height - height * scale) / 2) - minimumY * scale

            context.translateBy(x: offsetX, y: offsetY)
            context.scaleBy(x: scale, y: scale)
            for stroke in paths where stroke.count > 1 {
                var path = Path()
                path.move(to: stroke[0])
                for point in stroke.dropFirst() {
                    path.addLine(to: point)
                }
                context.stroke(
                    path,
                    with: .color(.primary),
                    style: StrokeStyle(
                        lineWidth: max(1 / scale, 0.6),
                        lineCap: .round,
                        lineJoin: .round
                    )
                )
            }
        }
        .accessibilityHidden(true)
    }
}

private enum HandwritingStylePreviewRepository {
    static func paths(for styleID: Int) -> [[CGPoint]] {
        let store = HandwritingStyleStore()
        guard let sample = try? store.load(id: styleID) else { return [] }
        var x = 0.0
        var y = 0.0
        var currentStroke = [CGPoint]()
        var strokes = [[CGPoint]]()

        for offset in sample.strokes {
            x += Double(offset.x)
            y -= Double(offset.y)
            currentStroke.append(CGPoint(x: x, y: y))
            if offset.penUp == 1 {
                if currentStroke.count > 1 {
                    strokes.append(currentStroke)
                }
                currentStroke = []
            }
        }
        if currentStroke.count > 1 {
            strokes.append(currentStroke)
        }
        return strokes
    }
}
