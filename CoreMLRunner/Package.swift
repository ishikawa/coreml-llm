// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "CoreMLRunner",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .executable(name: "coreml-runner", targets: ["CoreMLRunnerCLI", "CoreMLRunner"]),
        .library(name: "CoreMLRunner", targets: ["CoreMLRunner"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            branch: "preview"),
        .package(url: "https://github.com/apple/swift-argument-parser", from: "1.3.0"),
    ],
    targets: [
        .target(
            name: "CoreMLRunner",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            // NOTE: .process() を使うとリソースの名前が変わるのか、bundle から取得できない
            resources: [.copy("./gpt2-baseline.mlmodelc")]),
        .executableTarget(
            name: "CoreMLRunnerCLI",
            dependencies: [
                "CoreMLRunner",
                .product(name: "ArgumentParser", package: "swift-argument-parser"),
            ],
            swiftSettings: [
                .unsafeFlags(["-parse-as-library"])
            ]
        ),
    ]
)
