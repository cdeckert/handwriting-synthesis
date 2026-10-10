import SwiftUI

struct SettingsView: View {
    @Environment(\.dismiss) private var dismiss

    var body: some View {
        NavigationStack {
            Form {
                Section("Privacy & performance") {
                    Label("Generated privately on this device", systemImage: "checkmark.shield.fill")
                        .foregroundStyle(.green)

                    LabeledContent("Model", value: "Core ML")
                    LabeledContent("Connection", value: "Not required")

                    Text("Your text, handwriting styles, and generated pages are not uploaded to a server.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Language support") {
                    Label("Latin alphabet", systemImage: "character.cursor.ibeam")
                    Text("Includes German umlauts, ß, and French accented letters.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }

                Section("Export") {
                    Label("Scalable SVG", systemImage: "square.and.arrow.up")
                    Text("SVG files stay sharp at any size and can be opened in most design and illustration apps.")
                        .font(.footnote)
                        .foregroundStyle(.secondary)
                }
            }
            .navigationTitle("About Handwriting")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .confirmationAction) {
                    Button("Done") { dismiss() }
                }
            }
        }
    }
}
