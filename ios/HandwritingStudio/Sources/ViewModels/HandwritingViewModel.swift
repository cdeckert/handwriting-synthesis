import Foundation

@MainActor
final class HandwritingViewModel: ObservableObject {
    static let maximumTextLength = 911

    @Published var text = "Hello World!\nWelcome to Handwriting Studio"
    @Published var styles = HandwritingStyle.bundled
        + CustomHandwritingStyleRepository.styles()
    @Published var selectedStyleID: Int? = 9
    @Published var alignment: TextAlignment = .center
    @Published var pageFormat: PageFormat = .a4
    @Published var pageOrientation: PageOrientation = .portrait
    @Published var fontSize = 36.0
    @Published var bias = 0.75
    @Published var document: RenderDocument?
    @Published var isRendering = false
    @Published var generationProgress = 0.0
    @Published var isExporting = false
    @Published var errorMessage: String?

    private let generationService = HandwritingGenerationService()
    private var previewTask: Task<Void, Never>?
    private var activeRenderID: UUID?
    private var renderedRequest: GenerationRequest?
    private var renderedBias: Double?

    deinit {
        previewTask?.cancel()
    }

    func start() {
        schedulePreview(immediately: true)
    }

    func schedulePreview(immediately: Bool = false) {
        previewTask?.cancel()

        guard !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else {
            document = nil
            errorMessage = nil
            isRendering = false
            return
        }

        previewTask = Task { [weak self] in
            if !immediately {
                try? await Task.sleep(for: .milliseconds(450))
            }
            guard !Task.isCancelled else { return }
            await self?.renderPreview()
        }
    }

    func renderPreview() async {
        let renderID = UUID()
        let request = generationRequest
        let requestedBias = bias
        activeRenderID = renderID
        isRendering = true
        generationProgress = 0
        defer {
            if activeRenderID == renderID {
                isRendering = false
            }
        }

        do {
            let rendered = try await generationService.render(
                request: request,
                bias: requestedBias,
                progress: { [weak self] value in
                    Task { @MainActor [weak self] in
                        guard self?.activeRenderID == renderID else { return }
                        self?.generationProgress = value
                    }
                }
            )
            try Task.checkCancellation()
            guard request == generationRequest, requestedBias == bias else { return }
            document = rendered
            generationProgress = 1
            renderedRequest = request
            renderedBias = requestedBias
            errorMessage = nil
        } catch is CancellationError {
            return
        } catch {
            document = nil
            errorMessage = error.localizedDescription
        }
    }

    func exportSVG() async -> SharedFile? {
        isExporting = true
        defer { isExporting = false }

        do {
            let rendered: RenderDocument
            let request = generationRequest
            let requestedBias = bias
            if
                let document,
                renderedRequest == request,
                renderedBias == requestedBias
            {
                rendered = document
            } else {
                rendered = try await generationService.render(
                    request: request,
                    bias: requestedBias
                )
                document = rendered
                renderedRequest = request
                renderedBias = requestedBias
            }

            let directory = FileManager.default.temporaryDirectory
                .appending(path: "HandwritingStudio", directoryHint: .isDirectory)
            try FileManager.default.createDirectory(
                at: directory,
                withIntermediateDirectories: true
            )
            let fileURL = directory.appending(
                path: "handwriting-\(Int(Date().timeIntervalSince1970)).svg"
            )
            try NativeSVGRenderer.data(for: rendered).write(
                to: fileURL,
                options: .atomic
            )
            errorMessage = nil
            return SharedFile(url: fileURL)
        } catch {
            errorMessage = error.localizedDescription
            return nil
        }
    }

    func savePersonalStyle(
        name: String,
        characters: String,
        strokes: [StrokeOffset]
    ) throws {
        let style = try CustomHandwritingStyleRepository.save(
            name: name,
            characters: characters,
            strokes: strokes
        )
        styles = HandwritingStyle.bundled
            + CustomHandwritingStyleRepository.styles()
        selectedStyleID = style.id
        errorMessage = nil
        schedulePreview(immediately: true)
    }

    var pageLayout: HandwritingPageLayout {
        HandwritingPageLayout(
            format: pageFormat,
            orientation: pageOrientation,
            fontSize: fontSize
        )
    }

    private var generationRequest: GenerationRequest {
        GenerationRequest(
            text: text,
            style: selectedStyleID,
            alignment: alignment,
            pageFormat: pageFormat,
            pageOrientation: pageOrientation,
            fontSize: fontSize
        )
    }
}
