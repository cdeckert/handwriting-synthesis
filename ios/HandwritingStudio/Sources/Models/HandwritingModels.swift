import Foundation

struct HandwritingStyle: Identifiable, Equatable {
    let id: Int
    let label: String
    let detail: String

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

struct GenerationRequest: Encodable, Equatable {
    let text: String
    let style: Int?
    let alignment: TextAlignment
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
}

struct SharedFile: Identifiable {
    let id = UUID()
    let url: URL
}
