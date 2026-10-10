import Foundation

struct HandwritingStyle: Identifiable, Equatable {
    let id: Int
    let label: String
    let detail: String
    let isCustom: Bool

    init(id: Int, label: String, detail: String, isCustom: Bool = false) {
        self.id = id
        self.label = label
        self.detail = detail
        self.isCustom = isCustom
    }

    static let bundled = [
        HandwritingStyle(id: 0, label: "Light Loops", detail: "Thin and relaxed"),
        HandwritingStyle(id: 1, label: "Wide Print", detail: "Open and geometric"),
        HandwritingStyle(id: 2, label: "Compact Script", detail: "Small and connected"),
        HandwritingStyle(id: 3, label: "Tall Flourish", detail: "High, expressive strokes"),
        HandwritingStyle(id: 4, label: "Rounded Notes", detail: "Soft and spacious"),
        HandwritingStyle(id: 5, label: "Fine Print", detail: "Neat and understated"),
        HandwritingStyle(id: 6, label: "Casual Loops", detail: "Loose and personal"),
        HandwritingStyle(id: 7, label: "Open Round", detail: "Clear and generous"),
        HandwritingStyle(id: 8, label: "Elegant Narrow", detail: "Tall and refined"),
        HandwritingStyle(id: 9, label: "Bold Slant", detail: "Confident and fluid"),
        HandwritingStyle(id: 10, label: "Playful Script", detail: "Bouncy and informal"),
        HandwritingStyle(id: 11, label: "Fine Cursive", detail: "Small and flowing"),
        HandwritingStyle(id: 12, label: "Airy Script", detail: "Light and spacious"),
    ]
}

enum TextAlignment: String, CaseIterable, Codable, Identifiable {
    case left
    case center

    var id: Self { self }

    var title: String {
        switch self {
        case .left: "Left"
        case .center: "Center"
        }
    }

    var symbol: String {
        switch self {
        case .left: "text.alignleft"
        case .center: "text.aligncenter"
        }
    }
}

enum PageOrientation: String, CaseIterable, Codable, Identifiable {
    case portrait
    case landscape

    var id: Self { self }

    var title: String {
        switch self {
        case .portrait: "Portrait"
        case .landscape: "Landscape"
        }
    }
}

enum PageFormat: String, CaseIterable, Codable, Identifiable {
    case a5
    case a4
    case a3
    case letter
    case legal
    case phone
    case tablet
    case desktopHD
    case square

    var id: Self { self }

    static let paperFormats: [PageFormat] = [.a5, .a4, .a3, .letter, .legal]
    static let screenFormats: [PageFormat] = [.phone, .tablet, .desktopHD, .square]

    var title: String {
        switch self {
        case .a5: "A5"
        case .a4: "A4"
        case .a3: "A3"
        case .letter: "Letter"
        case .legal: "Legal"
        case .phone: "Phone"
        case .tablet: "Tablet"
        case .desktopHD: "Desktop HD"
        case .square: "Square"
        }
    }

    var detail: String {
        switch self {
        case .a5: "148 × 210 mm"
        case .a4: "210 × 297 mm"
        case .a3: "297 × 420 mm"
        case .letter: "8.5 × 11 in"
        case .legal: "8.5 × 14 in"
        case .phone: "390 × 844 px"
        case .tablet: "1024 × 1366 px"
        case .desktopHD: "1920 × 1080 px"
        case .square: "1080 × 1080 px"
        }
    }

    var defaultOrientation: PageOrientation {
        self == .desktopHD ? .landscape : .portrait
    }

    var isPaper: Bool {
        Self.paperFormats.contains(self)
    }

    func dimensions(orientation: PageOrientation) -> PageDimensions {
        let natural: PageDimensions = switch self {
        case .a5: PageDimensions(width: 419.53, height: 595.28)
        case .a4: PageDimensions(width: 595.28, height: 841.89)
        case .a3: PageDimensions(width: 841.89, height: 1190.55)
        case .letter: PageDimensions(width: 612, height: 792)
        case .legal: PageDimensions(width: 612, height: 1008)
        case .phone: PageDimensions(width: 390, height: 844)
        case .tablet: PageDimensions(width: 1024, height: 1366)
        case .desktopHD: PageDimensions(width: 1920, height: 1080)
        case .square: PageDimensions(width: 1080, height: 1080)
        }

        guard natural.width != natural.height else { return natural }
        let shortEdge = min(natural.width, natural.height)
        let longEdge = max(natural.width, natural.height)
        switch orientation {
        case .portrait:
            return PageDimensions(width: shortEdge, height: longEdge)
        case .landscape:
            return PageDimensions(width: longEdge, height: shortEdge)
        }
    }
}

struct PageDimensions: Codable, Equatable {
    let width: Double
    let height: Double
}

struct GenerationRequest: Encodable, Equatable {
    let text: String
    let style: Int?
    let alignment: TextAlignment
    let pageFormat: PageFormat
    let pageOrientation: PageOrientation
    let fontSize: Double
}

struct RenderPoint: Codable, Equatable {
    let x: Double
    let y: Double
    let move: Bool
}

struct RenderedPath: Codable, Equatable {
    let strokeColor: String
    let lineWidth: Double
    let points: [RenderPoint]
}

struct RenderDocument: Codable, Equatable {
    let width: Double
    let height: Double
    let backgroundColor: String
    let paths: [RenderedPath]
    let unit: CanvasUnit?
}

enum CanvasUnit: String, Codable {
    case points = "pt"
    case pixels = "px"
}

struct SharedFile: Identifiable {
    let id = UUID()
    let url: URL
}

enum ExportFormat: String, CaseIterable, Identifiable {
    case svg
    case pdf
    case png
    case jpeg

    var id: Self { self }

    var title: String {
        switch self {
        case .svg: "SVG"
        case .pdf: "PDF"
        case .png: "PNG"
        case .jpeg: "JPG"
        }
    }

    var detail: String {
        switch self {
        case .svg: "Scalable vector for editing"
        case .pdf: "Print-ready document"
        case .png: "Lossless image"
        case .jpeg: "Compact image"
        }
    }

    var systemImage: String {
        switch self {
        case .svg: "point.3.connected.trianglepath.dotted"
        case .pdf: "doc.richtext"
        case .png: "photo"
        case .jpeg: "photo.fill"
        }
    }

    var fileExtension: String {
        switch self {
        case .svg: "svg"
        case .pdf: "pdf"
        case .png: "png"
        case .jpeg: "jpg"
        }
    }
}
