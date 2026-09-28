import SwiftUI

struct HandwritingStylePicker: View {
    let styles: [HandwritingStyle]
    @Binding var selection: Int?
    @State private var isExpanded = false

    private var selectedStyle: HandwritingStyle {
        styles.first(where: { $0.id == selection }) ?? styles[0]
    }

    var body: some View {
        VStack(spacing: 0) {
            Button {
                withAnimation(.snappy) {
                    isExpanded.toggle()
                }
            } label: {
                HStack(spacing: 12) {
                    styleDescription(selectedStyle)
                    Spacer(minLength: 8)
                    HandwritingStyleSampleView(styleID: selectedStyle.id)
                        .frame(width: 128, height: 36)
                    Image(systemName: "chevron.down")
                        .font(.caption.weight(.semibold))
                        .rotationEffect(.degrees(isExpanded ? 180 : 0))
                        .foregroundStyle(.secondary)
                }
                .contentShape(Rectangle())
                .padding(12)
            }
            .buttonStyle(.plain)
            .accessibilityLabel("Writing style, \(selectedStyle.label)")
            .accessibilityHint("Shows all writing styles")

            if isExpanded {
                Divider()
                ScrollView {
                    LazyVStack(spacing: 0) {
                        ForEach(styles) { style in
                            Button {
                                selection = style.id
                                withAnimation(.snappy) {
                                    isExpanded = false
                                }
                            } label: {
                                HStack(spacing: 12) {
                                    styleDescription(style)
                                    Spacer(minLength: 8)
                                    HandwritingStyleSampleView(styleID: style.id)
                                        .frame(width: 150, height: 38)
                                    Image(systemName: selection == style.id ? "checkmark.circle.fill" : "circle")
                                        .foregroundStyle(selection == style.id ? Color.accentColor : .secondary)
                                }
                                .contentShape(Rectangle())
                                .padding(.horizontal, 12)
                                .padding(.vertical, 8)
                            }
                            .buttonStyle(.plain)

                            if style.id != styles.last?.id {
                                Divider().padding(.leading, 12)
                            }
                        }
                    }
                }
                .frame(maxHeight: 330)
            }
        }
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 12, style: .continuous)
                .stroke(.quaternary)
        }
    }

    private func styleDescription(_ style: HandwritingStyle) -> some View {
        VStack(alignment: .leading, spacing: 2) {
            Text(style.label)
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.primary)
            Text(style.detail)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)
        }
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
