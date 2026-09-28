import Foundation

enum CustomHandwritingStyleRepository {
    private static let firstCustomID = 1_000

    static func styles() -> [HandwritingStyle] {
        (try? records())?.map(\.style) ?? []
    }

    static func sample(id: Int) throws -> HandwritingStyleSample? {
        guard id >= firstCustomID else { return nil }
        guard let record = try records().first(where: { $0.id == id }) else {
            return nil
        }
        return HandwritingStyleSample(
            characters: record.characters,
            strokes: record.strokes
        )
    }

    @discardableResult
    static func save(
        name: String,
        characters: String,
        strokes: [StrokeOffset]
    ) throws -> HandwritingStyle {
        guard !characters.isEmpty, !strokes.isEmpty else {
            throw NativeHandwritingError.customStyleStorage
        }

        var stored = try records()
        let id = max(stored.map(\.id).max() ?? (firstCustomID - 1), firstCustomID - 1) + 1
        let trimmedName = name.trimmingCharacters(in: .whitespacesAndNewlines)
        let record = StoredCustomHandwritingStyle(
            id: id,
            label: trimmedName.isEmpty ? "My handwriting" : trimmedName,
            characters: characters,
            strokes: strokes,
            createdAt: Date()
        )
        stored.append(record)
        try write(stored)
        return record.style
    }

    private static func records() throws -> [StoredCustomHandwritingStyle] {
        let url = try storageURL(createDirectory: false)
        guard FileManager.default.fileExists(atPath: url.path) else { return [] }
        do {
            return try JSONDecoder().decode(
                [StoredCustomHandwritingStyle].self,
                from: Data(contentsOf: url)
            )
        } catch {
            throw NativeHandwritingError.customStyleStorage
        }
    }

    private static func write(_ styles: [StoredCustomHandwritingStyle]) throws {
        let url = try storageURL(createDirectory: true)
        do {
            let encoder = JSONEncoder()
            encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try encoder.encode(styles).write(to: url, options: .atomic)
        } catch {
            throw NativeHandwritingError.customStyleStorage
        }
    }

    private static func storageURL(createDirectory: Bool) throws -> URL {
        do {
            let applicationSupport = try FileManager.default.url(
                for: .applicationSupportDirectory,
                in: .userDomainMask,
                appropriateFor: nil,
                create: createDirectory
            )
            let directory = applicationSupport.appending(
                path: "PersonalHandwritingStyles",
                directoryHint: .isDirectory
            )
            if createDirectory {
                try FileManager.default.createDirectory(
                    at: directory,
                    withIntermediateDirectories: true
                )
            }
            return directory.appending(path: "styles.json")
        } catch {
            throw NativeHandwritingError.customStyleStorage
        }
    }
}

private struct StoredCustomHandwritingStyle: Codable {
    let id: Int
    let label: String
    let characters: String
    let strokes: [StrokeOffset]
    let createdAt: Date

    var style: HandwritingStyle {
        HandwritingStyle(
            id: id,
            label: label,
            detail: "Personal Apple Pencil style",
            isCustom: true
        )
    }
}
