// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "GPT2Model",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .executable(name: "gpt2-runner", targets: ["GPT2ModelCLI", "GPT2Model"]),
        .library(name: "GPT2Model", targets: ["GPT2Model"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            branch: "preview")
    ],
    targets: [
        .target(
            name: "GPT2Model",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            // NOTE: .process() を使うとリソースの名前が変わるのか、bundle から取得できない
            resources: [.copy("./GPT2Model.mlmodelc")]),
        .executableTarget(
            name: "GPT2ModelCLI",
            dependencies: ["GPT2Model"]),
    ]
)
