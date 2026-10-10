import SwiftUI

struct ContentView: View {
    @StateObject private var viewModel = HandwritingViewModel()
    @State private var selectedInspectorSection = InspectorSection.text
    @State private var isShowingSettings = false
    @State private var isShowingPersonalStyleCapture = false
    @State private var isShowingFullPreview = false
    @State private var isShowingExportFormats = false
    @State private var sharedFile: SharedFile?
    @FocusState private var isTextEditorFocused: Bool

    var body: some View {
        NavigationStack {
            GeometryReader { geometry in
                Group {
                    if geometry.size.width >= 820 {
                        regularLayout(availableWidth: geometry.size.width)
                    } else {
                        compactLayout(availableHeight: geometry.size.height)
                    }
                }
                .frame(
                    width: geometry.size.width,
                    height: geometry.size.height,
                    alignment: .top
                )
            }
            .background(workspaceBackground)
            .navigationTitle("Handwriting Studio")
            .navigationBarTitleDisplayMode(.inline)
            .toolbar {
                ToolbarItem(placement: .topBarLeading) {
                    generationStatus
                }
                ToolbarItem(placement: .topBarTrailing) {
                    Button("Settings", systemImage: "gearshape") {
                        isShowingSettings = true
                    }
                }
            }
        }
        .task { viewModel.start() }
        .onChange(of: viewModel.text) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.selectedStyleID) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.alignment) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.pageFormat) { _, format in
            viewModel.pageOrientation = format.defaultOrientation
            viewModel.schedulePreview()
        }
        .onChange(of: viewModel.pageOrientation) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.fontSize) { _, _ in viewModel.schedulePreview() }
        .onChange(of: viewModel.bias) { _, _ in viewModel.schedulePreview() }
        .sheet(isPresented: $isShowingSettings) {
            SettingsView()
        }
        .sheet(isPresented: $isShowingPersonalStyleCapture) {
            PersonalHandwritingCaptureView { name, characters, strokes in
                try viewModel.savePersonalStyle(
                    name: name,
                    characters: characters,
                    strokes: strokes
                )
            }
        }
        .sheet(item: $sharedFile) { file in
            ShareSheet(items: [file.url])
        }
        .confirmationDialog(
            "Choose export format",
            isPresented: $isShowingExportFormats,
            titleVisibility: .visible
        ) {
            ForEach(ExportFormat.allCases) { format in
                Button {
                    export(format)
                } label: {
                    Label(format.title, systemImage: format.systemImage)
                }
            }
        } message: {
            Text("The selected file will be prepared and opened in the share sheet.")
        }
        .fullScreenCover(isPresented: $isShowingFullPreview) {
            if let document = viewModel.document {
                FullPreviewView(document: document)
            }
        }
    }

    private var workspaceBackground: Color {
        Color(uiColor: .secondarySystemGroupedBackground)
    }

    private func regularLayout(availableWidth: CGFloat) -> some View {
        let inspectorWidth = min(max(availableWidth * 0.32, 360), 440)

        return HStack(spacing: 0) {
            ipadCanvasWorkspace
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            Divider()

            ipadInspector
                .frame(width: inspectorWidth)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
        }
    }

    private var ipadCanvasWorkspace: some View {
        VStack(spacing: 0) {
            HStack(spacing: 12) {
                VStack(alignment: .leading, spacing: 3) {
                    Text("Preview")
                        .font(.headline)
                    Label(pageSummary, systemImage: "doc")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer()

                if viewModel.isRendering {
                    renderingPill
                }

                Button("Full Screen", systemImage: "arrow.up.left.and.arrow.down.right") {
                    isShowingFullPreview = true
                }
                .buttonStyle(.bordered)
                .disabled(viewModel.document == nil)
            }
            .padding(.horizontal, 24)
            .padding(.vertical, 16)
            .background(Color(uiColor: .systemBackground))

            ZStack {
                Color(uiColor: .systemGroupedBackground)

                if let document = viewModel.document {
                    StableDocumentPreview(
                        document: document,
                        contentInsets: EdgeInsets(
                            top: 40,
                            leading: 40,
                            bottom: 40,
                            trailing: 40
                        )
                    )
                        .transition(.opacity)
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
                        description: Text("Your page will appear here as you type.")
                    )
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .animation(.easeInOut(duration: 0.2), value: viewModel.document)
        }
    }

    private var ipadInspector: some View {
        VStack(spacing: 0) {
            ScrollView {
                LazyVStack(spacing: 14) {
                    InspectorSectionCard(
                        title: "Text",
                        subtitle: "What should the page say?",
                        systemImage: "text.cursor"
                    ) {
                        textInspector
                    }

                    InspectorSectionCard(
                        title: "Handwriting",
                        subtitle: "Choose a style and its character",
                        systemImage: "scribble.variable"
                    ) {
                        styleInspector
                    }

                    InspectorSectionCard(
                        title: "Page",
                        subtitle: "Format, size and alignment",
                        systemImage: "doc"
                    ) {
                        pageInspector
                    }
                }
                .padding(16)
            }
            .scrollDismissesKeyboard(.interactively)

            Divider()

            ipadExportBar
        }
    }

    private var ipadExportBar: some View {
        VStack(alignment: .leading, spacing: 8) {
            Menu {
                ForEach(ExportFormat.allCases) { format in
                    Button {
                        export(format)
                    } label: {
                        Label(format.title, systemImage: format.systemImage)
                    }
                }
            } label: {
                HStack(spacing: 8) {
                    if viewModel.isExporting {
                        ProgressView()
                            .tint(.white)
                    } else {
                        Image(systemName: "square.and.arrow.up")
                    }
                    Text(viewModel.isExporting ? "Preparing…" : "Export & Share")
                        .fontWeight(.semibold)
                    Spacer()
                    Image(systemName: "chevron.up.chevron.down")
                        .font(.caption2.weight(.bold))
                        .opacity(0.75)
                }
                .frame(maxWidth: .infinity)
                .padding(.vertical, 12)
                .padding(.horizontal, 14)
            }
            .buttonStyle(.borderedProminent)
            .disabled(
                viewModel.isExporting
                    || viewModel.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
            )

            Text("SVG, PDF, PNG or JPG")
                .font(.caption2)
                .foregroundStyle(.secondary)
        }
        .padding(16)
        .background(Color(uiColor: .systemBackground))
    }

    private func compactLayout(availableHeight: CGFloat) -> some View {
        let inspectorHeight = min(max(330, availableHeight * 0.47), 382)

        return VStack(spacing: 0) {
            canvasWorkspace
                .frame(maxWidth: .infinity, maxHeight: .infinity)

            inspector(showsGrabber: true)
                .frame(height: inspectorHeight)
                .background(Color(uiColor: .systemBackground))
                .clipShape(
                    UnevenRoundedRectangle(
                        topLeadingRadius: 24,
                        topTrailingRadius: 24
                    )
                )
                .shadow(color: .black.opacity(0.08), radius: 18, y: -4)
        }
    }

    private var canvasWorkspace: some View {
        VStack(spacing: 0) {
            HStack(spacing: 8) {
                Label(pageSummary, systemImage: "doc")
                    .lineLimit(1)

                Spacer()

                Button("Full screen", systemImage: "arrow.up.left.and.arrow.down.right") {
                    isShowingFullPreview = true
                }
                .labelStyle(.iconOnly)
                .buttonStyle(.bordered)
                .buttonBorderShape(.circle)
                .disabled(viewModel.document == nil)
            }
            .font(.caption)
            .foregroundStyle(.secondary)
            .padding(.horizontal, 16)
            .padding(.vertical, 8)

            ZStack {
                if let document = viewModel.document {
                    StableDocumentPreview(
                        document: document,
                        contentInsets: EdgeInsets(
                            top: 0,
                            leading: 30,
                            bottom: 14,
                            trailing: 30
                        )
                    )
                        .transition(.opacity)
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
                        description: Text("Your page will appear here as you type.")
                    )
                }

                if viewModel.isRendering, viewModel.document != nil {
                    VStack {
                        Spacer()
                        renderingPill
                    }
                    .padding(.bottom, 24)
                    .transition(.opacity.combined(with: .move(edge: .bottom)))
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity)
            .animation(.easeInOut(duration: 0.2), value: viewModel.document)
        }
    }

    private var pageSummary: String {
        "\(viewModel.pageFormat.title) · \(viewModel.pageOrientation.title)"
    }

    private var generationStatus: some View {
        Label(
            viewModel.isRendering ? "Generating" : "On device",
            systemImage: viewModel.isRendering ? "sparkles" : "checkmark.circle.fill"
        )
        .font(.caption)
        .foregroundStyle(viewModel.isRendering ? Color.secondary : Color.green)
        .accessibilityLabel(
            viewModel.isRendering
                ? "Generating handwriting on this device"
                : "Ready, generation happens on this device"
        )
    }

    private var renderingPill: some View {
        HStack(spacing: 9) {
            ProgressView(value: viewModel.generationProgress)
                .frame(width: 72)
                .tint(.accentColor)
            Text(viewModel.generationProgress, format: .percent.precision(.fractionLength(0)))
                .monospacedDigit()
        }
        .font(.caption)
        .padding(.horizontal, 14)
        .padding(.vertical, 9)
        .background(.regularMaterial, in: Capsule())
        .shadow(color: .black.opacity(0.08), radius: 8, y: 3)
        .accessibilityElement(children: .combine)
        .accessibilityLabel("Generating handwriting")
    }

    private func inspector(showsGrabber: Bool) -> some View {
        VStack(spacing: 0) {
            if showsGrabber {
                Capsule()
                    .fill(.quaternary)
                    .frame(width: 34, height: 4)
                    .padding(.top, 8)
                    .padding(.bottom, 10)
                    .accessibilityHidden(true)
            }

            Picker("Editor", selection: $selectedInspectorSection) {
                ForEach(InspectorSection.allCases) { section in
                    Label(section.title, systemImage: section.symbol)
                        .tag(section)
                }
            }
            .pickerStyle(.segmented)
            .padding(.horizontal, 16)
            .padding(.top, showsGrabber ? 0 : 16)

            Group {
                switch selectedInspectorSection {
                case .text:
                    textInspector
                case .style:
                    styleInspector
                case .page:
                    pageInspector
                }
            }
            .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .top)
            .padding(.horizontal, 16)
            .padding(.top, 12)

            Divider()

            exportBar
        }
    }

    private var textInspector: some View {
        VStack(spacing: 7) {
            TextEditor(text: $viewModel.text)
                .focused($isTextEditorFocused)
                .font(.body)
                .frame(minHeight: 100, maxHeight: 260)
                .padding(8)
                .scrollContentBackground(.hidden)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .clipShape(RoundedRectangle(cornerRadius: 14, style: .continuous))
                .overlay {
                    RoundedRectangle(cornerRadius: 14, style: .continuous)
                        .stroke(.quaternary)
                }
                .accessibilityLabel("Text to turn into handwriting")

            HStack {
                if isTextEditorFocused {
                    Button("Done") {
                        isTextEditorFocused = false
                    }
                    .fontWeight(.semibold)
                } else {
                    Text("Preview updates automatically")
                }
                Spacer()
                Text("\(viewModel.text.count)/\(HandwritingViewModel.maximumTextLength)")
                    .monospacedDigit()
            }
            .font(.caption2)
            .foregroundStyle(.secondary)
        }
    }

    private var styleInspector: some View {
        VStack(alignment: .leading, spacing: 10) {
            HandwritingStylePicker(
                styles: viewModel.styles,
                selection: $viewModel.selectedStyleID
            )

            HStack {
                Text("Writing character")
                    .font(.subheadline.weight(.medium))
                Spacer()
                Text(characterLabel)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            Slider(value: $viewModel.bias, in: 0 ... 1, step: 0.05) {
                Text("Writing character")
            } minimumValueLabel: {
                Text("Natural").font(.caption2)
            } maximumValueLabel: {
                Text("Regular").font(.caption2)
            }

            if UIDevice.current.userInterfaceIdiom == .pad {
                Button("Create a style from my handwriting", systemImage: "applepencil") {
                    isShowingPersonalStyleCapture = true
                }
                .buttonStyle(.bordered)
            }
        }
    }

    private var characterLabel: String {
        switch viewModel.bias {
        case ..<0.34: "Natural"
        case ..<0.67: "Balanced"
        default: "Regular"
        }
    }

    private var pageInspector: some View {
        VStack(spacing: 10) {
            HStack(spacing: 12) {
                Menu {
                    Section("Paper") {
                        ForEach(PageFormat.paperFormats) { format in
                            Button {
                                viewModel.pageFormat = format
                            } label: {
                                formatLabel(format)
                            }
                        }
                    }
                    Section("Screens") {
                        ForEach(PageFormat.screenFormats) { format in
                            Button {
                                viewModel.pageFormat = format
                            } label: {
                                formatLabel(format)
                            }
                        }
                    }
                } label: {
                    InspectorValueCard(
                        title: "Format",
                        value: viewModel.pageFormat.title,
                        detail: viewModel.pageFormat.detail,
                        systemImage: "doc"
                    )
                }
                .buttonStyle(.plain)

                InspectorValueCard(
                    title: "Writing size",
                    value: "\(Int(viewModel.fontSize)) pt",
                    detail: "\(viewModel.pageLayout.maximumCharactersPerLine) chars/line",
                    systemImage: "textformat.size"
                )
            }

            Slider(value: $viewModel.fontSize, in: 20 ... 64, step: 2) {
                Text("Writing size")
            } minimumValueLabel: {
                Text("A").font(.caption2)
            } maximumValueLabel: {
                Text("A").font(.title3)
            }

            HStack(spacing: 10) {
                Picker("Orientation", selection: $viewModel.pageOrientation) {
                    ForEach(PageOrientation.allCases) { orientation in
                        Text(orientation.title).tag(orientation)
                    }
                }
                .pickerStyle(.segmented)

                Picker("Alignment", selection: $viewModel.alignment) {
                    ForEach(TextAlignment.allCases) { alignment in
                        Image(systemName: alignment.symbol).tag(alignment)
                    }
                }
                .pickerStyle(.segmented)
                .frame(maxWidth: 126)
            }
        }
    }

    @ViewBuilder
    private func formatLabel(_ format: PageFormat) -> some View {
        if viewModel.pageFormat == format {
            Label(format.title, systemImage: "checkmark")
        } else {
            Text(format.title)
        }
    }

    private var exportBar: some View {
        Button {
            isShowingExportFormats = true
        } label: {
            HStack(spacing: 8) {
                if viewModel.isExporting {
                    ProgressView()
                        .tint(.white)
                } else {
                    Image(systemName: "square.and.arrow.up")
                }
                Text(viewModel.isExporting ? "Preparing…" : "Export & Share")
                    .fontWeight(.semibold)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 12)
        }
        .buttonStyle(.borderedProminent)
        .disabled(
            viewModel.isExporting
                || viewModel.text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        )
        .padding(.horizontal, 16)
        .padding(.vertical, 10)
    }

    private func export(_ format: ExportFormat) {
        Task {
            sharedFile = await viewModel.export(format: format)
        }
    }
}

private enum InspectorSection: String, CaseIterable, Identifiable {
    case text
    case style
    case page

    var id: Self { self }

    var title: String {
        switch self {
        case .text: "Text"
        case .style: "Style"
        case .page: "Page"
        }
    }

    var symbol: String {
        switch self {
        case .text: "text.cursor"
        case .style: "scribble.variable"
        case .page: "doc"
        }
    }
}

private struct InspectorSectionCard<Content: View>: View {
    let title: String
    let subtitle: String
    let systemImage: String
    @ViewBuilder let content: Content

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(alignment: .top, spacing: 11) {
                Image(systemName: systemImage)
                    .font(.body.weight(.semibold))
                    .foregroundStyle(.tint)
                    .frame(width: 32, height: 32)
                    .background(Color.accentColor.opacity(0.12), in: RoundedRectangle(cornerRadius: 9))

                VStack(alignment: .leading, spacing: 2) {
                    Text(title)
                        .font(.headline)
                    Text(subtitle)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Spacer(minLength: 0)
            }

            content
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(16)
        .background(Color(uiColor: .systemBackground))
        .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
        .overlay {
            RoundedRectangle(cornerRadius: 18, style: .continuous)
                .stroke(.quaternary)
        }
    }
}

private struct InspectorValueCard: View {
    let title: String
    let value: String
    let detail: String
    let systemImage: String

    var body: some View {
        HStack(spacing: 10) {
            Image(systemName: systemImage)
                .foregroundStyle(.tint)
                .frame(width: 24)

            VStack(alignment: .leading, spacing: 2) {
                Text(title)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text(value)
                    .font(.subheadline.weight(.semibold))
                Text(detail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .lineLimit(1)
            }

            Spacer(minLength: 0)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(10)
        .background(Color(uiColor: .secondarySystemGroupedBackground))
        .clipShape(RoundedRectangle(cornerRadius: 13, style: .continuous))
    }
}

/// Keeps the paper at its keyboard-free size while allowing the surrounding
/// preview viewport to participate in normal keyboard avoidance.
private struct StableDocumentPreview: View {
    let document: RenderDocument
    let contentInsets: EdgeInsets

    @State private var referenceContainerSize: CGSize = .zero

    var body: some View {
        GeometryReader { geometry in
            let containerSize = referenceContainerSize == .zero
                ? geometry.size
                : referenceContainerSize
            let pageSize = fittedPageSize(in: containerSize)
            let pageOrigin = fittedPageOrigin(
                for: pageSize,
                in: containerSize
            )

            ScrollView(.vertical) {
                ZStack(alignment: .topLeading) {
                    HandwritingCanvas(document: document)
                        .frame(width: pageSize.width, height: pageSize.height)
                        .offset(x: pageOrigin.x, y: pageOrigin.y)
                }
                .frame(
                    width: geometry.size.width,
                    height: max(geometry.size.height, containerSize.height),
                    alignment: .topLeading
                )
            }
            .scrollIndicators(.hidden)
            .scrollDismissesKeyboard(.never)
            .scrollBounceBehavior(.basedOnSize)
            .onAppear {
                updateReferenceSize(with: geometry.size)
            }
            .onChange(of: geometry.size) { _, newSize in
                updateReferenceSize(with: newSize)
            }
        }
    }

    private func fittedPageSize(in containerSize: CGSize) -> CGSize {
        let availableWidth = max(
            1,
            containerSize.width - contentInsets.leading - contentInsets.trailing
        )
        let availableHeight = max(
            1,
            containerSize.height - contentInsets.top - contentInsets.bottom
        )
        let aspectRatio = document.width / document.height

        if availableWidth / availableHeight > aspectRatio {
            return CGSize(
                width: availableHeight * aspectRatio,
                height: availableHeight
            )
        }

        return CGSize(
            width: availableWidth,
            height: availableWidth / aspectRatio
        )
    }

    private func fittedPageOrigin(
        for pageSize: CGSize,
        in containerSize: CGSize
    ) -> CGPoint {
        let availableWidth = max(
            1,
            containerSize.width - contentInsets.leading - contentInsets.trailing
        )
        let availableHeight = max(
            1,
            containerSize.height - contentInsets.top - contentInsets.bottom
        )

        return CGPoint(
            x: contentInsets.leading + max(0, (availableWidth - pageSize.width) / 2),
            y: contentInsets.top + max(0, (availableHeight - pageSize.height) / 2)
        )
    }

    private func updateReferenceSize(with newSize: CGSize) {
        guard newSize.width > 0, newSize.height > 0 else { return }

        let widthChanged = abs(newSize.width - referenceContainerSize.width) > 1
        let grewVertically = newSize.height >= referenceContainerSize.height

        if referenceContainerSize == .zero || widthChanged || grewVertically {
            referenceContainerSize = newSize
        }
    }
}

private struct FullPreviewView: View {
    @Environment(\.dismiss) private var dismiss
    let document: RenderDocument

    var body: some View {
        NavigationStack {
            HandwritingCanvas(document: document)
                .aspectRatio(document.width / document.height, contentMode: .fit)
                .padding(24)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .background(Color(uiColor: .secondarySystemGroupedBackground))
                .navigationTitle("Preview")
                .navigationBarTitleDisplayMode(.inline)
                .toolbar {
                    ToolbarItem(placement: .confirmationAction) {
                        Button("Done") { dismiss() }
                    }
                }
        }
    }
}
