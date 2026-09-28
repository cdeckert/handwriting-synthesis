import PencilKit
import SwiftUI

struct PersonalHandwritingCaptureView: View {
    static let sampleText = "Waltz, bad nymph, for quick jigs vex."

    @Environment(\.dismiss) private var dismiss
    @State private var name = "My handwriting"
    @State private var drawing = PKDrawing()
    @State private var errorMessage: String?

    let onSave: (String, String, [StrokeOffset]) throws -> Void

    var body: some View {
        NavigationStack {
            Form {
                Section("Name") {
                    TextField("Style name", text: $name)
                }

                Section {
                    VStack(alignment: .leading, spacing: 12) {
                        Label("Write this sentence with Apple Pencil", systemImage: "applepencil")
                            .font(.headline)
                        Text(Self.sampleText)
                            .font(.title3)
                            .textSelection(.enabled)

                        ZStack(alignment: .bottomTrailing) {
                            PencilCanvasView(drawing: $drawing)
                                .frame(minHeight: 280)
                                .background(Color(uiColor: .systemBackground))

                            Button("Clear", systemImage: "eraser") {
                                drawing = PKDrawing()
                                errorMessage = nil
                            }
                            .buttonStyle(.bordered)
                            .padding(12)
                        }
                        .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                        .overlay {
                            RoundedRectangle(cornerRadius: 12, style: .continuous)
                                .stroke(.quaternary)
                        }

                        Text("Keep the sentence on one line when possible. The sample stays on this device and is used as a style guide for the model.")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                }

                if let errorMessage {
                    Section {
                        Label(errorMessage, systemImage: "exclamationmark.triangle")
                            .foregroundStyle(.red)
                    }
                }
            }
            .navigationTitle("My handwriting")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .cancellationAction) {
                    Button("Cancel") { dismiss() }
                }
                ToolbarItem(placement: .confirmationAction) {
                    Button("Save") { save() }
                        .fontWeight(.semibold)
                        .disabled(drawing.strokes.isEmpty)
                }
            }
        }
    }

    private func save() {
        do {
            let offsets = try PencilStrokeNormalizer.offsets(from: drawing)
            try onSave(name, Self.sampleText, offsets)
            dismiss()
        } catch {
            errorMessage = error.localizedDescription
        }
    }
}

private struct PencilCanvasView: UIViewRepresentable {
    @Binding var drawing: PKDrawing

    func makeCoordinator() -> Coordinator {
        Coordinator(drawing: $drawing)
    }

    func makeUIView(context: Context) -> PKCanvasView {
        let canvas = PKCanvasView()
        canvas.delegate = context.coordinator
        canvas.drawing = drawing
        canvas.drawingPolicy = UIDevice.current.userInterfaceIdiom == .pad
            ? .pencilOnly
            : .anyInput
        canvas.tool = PKInkingTool(.pen, color: .label, width: 3)
        canvas.backgroundColor = .secondarySystemGroupedBackground
        canvas.isOpaque = true
        canvas.alwaysBounceVertical = false
        canvas.alwaysBounceHorizontal = false
        return canvas
    }

    func updateUIView(_ canvas: PKCanvasView, context: Context) {
        guard canvas.drawing.dataRepresentation() != drawing.dataRepresentation() else {
            return
        }
        context.coordinator.isApplyingBinding = true
        canvas.drawing = drawing
        context.coordinator.isApplyingBinding = false
    }

    final class Coordinator: NSObject, PKCanvasViewDelegate {
        @Binding var drawing: PKDrawing
        var isApplyingBinding = false

        init(drawing: Binding<PKDrawing>) {
            _drawing = drawing
        }

        func canvasViewDrawingDidChange(_ canvasView: PKCanvasView) {
            guard !isApplyingBinding else { return }
            drawing = canvasView.drawing
        }
    }
}
