import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = HandwritingViewModel()
    @State private var isShowingSettings = false
    @State private var sharedFile: SharedFile?
    @FocusState private var isTextEditorFocused: Bool

    var body: some View {
        NavigationStack {
            ScrollView {
                VStack(spacing: 18) {
                    editorCard
                    previewCard
                }
                .padding(.horizontal)
                .padding(.bottom, 100)
            }
            .background(Color(uiColor: .systemGroupedBackground))
            .navigationTitle("Handwriting")
            .toolbar {
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Settings", systemImage: "gearshape") {
                        isShowingSettings = true
                    }
                }
                ToolbarItemGroup(placement: .keyboard) {
                    Spacer()
                    Button("Done") { isTextEditorFocused = false }
                }
            }
            .safeAreaInset(edge: .bottom) {
                exportBar
            }
        }
        .task { viewModel.start() }
        .onChange(of: viewModel.text) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.selectedStyleID) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.alignment) { _, _ in viewModel.schedulePreview() }
        .sheet(isPresented: $isShowingSettings) {
            SettingsView(viewModel: viewModel)
        }
        .sheet(item: $sharedFile) { file in
            ShareSheet(items: [file.url])
        }
    }

    private var editorCard: some View {
        VStack(alignment: .leading, spacing: 14) {
            Label("Your text", systemImage: "text.cursor")
                .font(.headline)

            TextEditor(text: $viewModel.text)
                .focused($isTextEditorFocused)
                .frame(minHeight: 150)
                .padding(8)
                .scrollContentBackground(.hidden)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 12, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                        .stroke(.quaternary)
                }

            HStack {
                Text("Up to 12 lines, 75 characters per line")
                Spacer()
                Text("\(viewModel.text.count)/\(HandwritingViewModel.maximumTextLength)")
                    .monospacedDigit()
            }
            .font(.caption)
            .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 7) {
                Text("Writing style")
                    .font(.subheadline.weight(.medium))
                HandwritingStylePicker(
                    styles: viewModel.styles,
                    selection: $viewModel.selectedStyleID
                )
            }

            Picker("Alignment", selection: $viewModel.alignment) {
                ForEach(TextAlignment.allCases) { alignment in
                    Label(alignment.title, systemImage: alignment.symbol).tag(alignment)
                }
            }
            .pickerStyle(.segmented)
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private var previewCard: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Label("Preview", systemImage: "pencil.and.outline")
                    .font(.headline)
                Label("On device", systemImage: "iphone.and.arrow.forward")
                    .font(.caption)
                    .foregroundStyle(.secondary)
                Spacer()
                if viewModel.isRendering {
                    Text(viewModel.generationProgress, format: .percent.precision(.fractionLength(0)))
                        .font(.caption.monospacedDigit())
                        .foregroundStyle(.secondary)
                }
            }

            if viewModel.isRendering {
                ProgressView(value: viewModel.generationProgress)
                    .progressViewStyle(.linear)
                    .tint(.accentColor)
                    .animation(.linear(duration: 0.12), value: viewModel.generationProgress)
                    .accessibilityLabel("Generating handwriting")
                    .accessibilityValue(
                        Text(viewModel.generationProgress, format: .percent)
                    )
            }

            Group {
                if let document = viewModel.document {
                    HandwritingCanvas(document: document)
                } else if let message = viewModel.errorMessage {
                    ContentUnavailableView(
                        "Preview unavailable",
                        systemImage: "exclamationmark.triangle",
                        description: Text(message)
                    )
                } else {
                    ContentUnavailableView(
                        "Start writing",
                        systemImage: "pencil.line",
                        description: Text("Your generated handwriting appears here.")
                    )
                }
            }
            .frame(maxWidth: .infinity, minHeight: 260)
        }
        .padding()
        .background(Color(uiColor: .systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 20, style: .continuous))
    }

    private var exportBar: some View {
        Button {
            Task { sharedFile = await viewModel.exportSVG() }
        } label: {
            HStack {
                if viewModel.isExporting {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "square.and.arrow.up")
                }
                Text(viewModel.isExporting ? "Preparing…" : "Share SVG")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 13)
        }
        .buttonStyle(.borderedProminent)
        .disabled(viewModel.isExporting || viewModel.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
        .padding(.horizontal)
        .padding(.vertical, 10)
        .background(.ultraThinMaterial)
    }
}
