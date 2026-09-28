// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "HandwritingCLI",
    platforms: [.macOS(.v14)],
    products: [
        .library(name: "HandwritingCore", targets: ["HandwritingCore"]),
        .executable(name: "handwriting-cli", targets: ["handwriting-cli"]),
    ],
    targets: [
        .target(
            name: "HandwritingCore",
            resources: [
                .copy("Resources/Models"),
                .copy("Resources/Styles"),
            ]
        ),
        .executableTarget(
            name: "handwriting-cli",
            dependencies: ["HandwritingCore"]
        ),
        .testTarget(
            name: "HandwritingCoreTests",
            dependencies: ["HandwritingCore"]
        ),
    ],
    swiftLanguageModes: [.v5]
)
