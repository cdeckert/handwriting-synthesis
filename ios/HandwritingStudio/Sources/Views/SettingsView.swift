import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: HandwritingViewModel
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Writing character") {
                    Slider(value: $viewModel.bias, in: 0 ... 1, step: 0.05) {
                        Text("Regularity")
                    } minimumValueLabel: {
                        Image(systemName: "scribble")
                    } maximumValueLabel: {
                        Image(systemName: "textformat")
                    }

                    LabeledContent("Regularity") {
                        Text(viewModel.bias, format: .number.precision(.fractionLength(2)))
                            .monospacedDigit()
                    }
                }

                Section("On-device generation") {
                    Label("Core ML model included", systemImage: "checkmark.seal.fill")
                        .foregroundStyle(.green)
                    Text("Handwriting is generated entirely on this iPhone. No network connection or server is required.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("Settings")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
        .onChange(of: viewModel.bias) { _, _ in
            viewModel.schedulePreview()
        }
    }
}
