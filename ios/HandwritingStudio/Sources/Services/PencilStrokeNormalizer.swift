import Foundation
import PencilKit

enum PencilStrokeNormalizer {
    private static let maximumPointCount = 1_200

    static func offsets(from drawing: PKDrawing) throws -> [StrokeOffset] {
        var capturedStrokes = drawing.strokes.compactMap { stroke -> [CGPoint]? in
            let points = stroke.path
                .interpolatedPoints(by: .distance(2.5))
                .map(\.location)
            return points.count > 1 ? points : nil
        }
        guard capturedStrokes.reduce(0, { $0 + $1.count }) >= 20 else {
            throw NativeHandwritingError.customStyleTooShort
        }

        let initialCount = capturedStrokes.reduce(0) { $0 + $1.count }
        if initialCount > maximumPointCount {
            let stride = Int(ceil(Double(initialCount) / Double(maximumPointCount)))
            capturedStrokes = capturedStrokes.map { points in
                var reduced = points.enumerated().compactMap { index, point in
                    index.isMultiple(of: stride) ? point : nil
                }
                if let last = points.last, reduced.last != last {
                    reduced.append(last)
                }
                return reduced
            }
        }

        var coordinates = [(x: Double, y: Double, penUp: Float)]()
        for points in capturedStrokes {
            for (index, point) in points.enumerated() {
                coordinates.append(
                    (
                        x: Double(point.x),
                        y: -Double(point.y),
                        penUp: index == points.count - 1 ? 1 : 0
                    )
                )
            }
        }
        guard coordinates.count >= 2 else {
            throw NativeHandwritingError.customStyleTooShort
        }

        coordinates = aligned(coordinates)
        var offsets = [StrokeOffset(x: 0, y: 0, penUp: 1)]
        offsets.reserveCapacity(coordinates.count)
        for index in 1 ..< coordinates.count {
            offsets.append(
                StrokeOffset(
                    x: Float(coordinates[index].x - coordinates[index - 1].x),
                    y: Float(coordinates[index].y - coordinates[index - 1].y),
                    penUp: coordinates[index].penUp
                )
            )
        }

        let lengths = offsets.dropFirst().compactMap { offset -> Double? in
            let length = hypot(Double(offset.x), Double(offset.y))
            return length.isFinite && length > 0.000_1 ? length : nil
        }.sorted()
        guard let median = lengths.isEmpty ? nil : lengths[lengths.count / 2] else {
            throw NativeHandwritingError.customStyleTooShort
        }

        return offsets.prefix(maximumPointCount).map {
            StrokeOffset(
                x: Float(Double($0.x) / median),
                y: Float(Double($0.y) / median),
                penUp: $0.penUp
            )
        }
    }

    private static func aligned(
        _ coordinates: [(x: Double, y: Double, penUp: Float)]
    ) -> [(x: Double, y: Double, penUp: Float)] {
        let count = Double(coordinates.count)
        let meanX = coordinates.reduce(0) { $0 + $1.x } / count
        let meanY = coordinates.reduce(0) { $0 + $1.y } / count
        let denominator = coordinates.reduce(0) {
            $0 + pow($1.x - meanX, 2)
        }
        guard denominator > 0 else { return coordinates }
        let slope = coordinates.reduce(0) {
            $0 + (($1.x - meanX) * ($1.y - meanY))
        } / denominator
        let angle = atan(slope)
        let cosine = cos(angle)
        let sine = sin(angle)

        return coordinates.map { point in
            let centeredX = point.x - meanX
            let centeredY = point.y - meanY
            return (
                x: (centeredX * cosine) - (centeredY * sine),
                y: (centeredX * sine) + (centeredY * cosine),
                penUp: point.penUp
            )
        }
    }
}
