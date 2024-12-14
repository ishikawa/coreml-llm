// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyCoreMLRunner",
    platforms: [
        .macOS(.v15),
        .iOS(.v18),
    ],
    products: [
        .executable(name: "my-core-ml-runner", targets: ["MyCoreMLRunner", "MyCoreMLModels"]),
        .library(name: "MyCoreMLModels", targets: ["MyCoreMLModels"]),
    ],
    dependencies: [
        .package(
            url: "https://github.com/huggingface/swift-transformers",
            branch: "preview")
    ],
    targets: [
        .target(
            name: "MyCoreMLModels",
            dependencies: [
                .product(name: "Transformers", package: "swift-transformers")
            ],
            path: "Sources",
            // NOTE: .process() を使うとリソースの名前が変わるのか、bundle から取得できない
            resources: [.copy("./GPT2Model.mlmodelc")]),
        .executableTarget(
            name: "MyCoreMLRunner",
            dependencies: ["MyCoreMLModels"],
            path: "cmd"),
    ]
)
