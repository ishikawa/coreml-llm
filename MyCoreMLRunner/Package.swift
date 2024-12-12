// swift-tools-version: 6.0
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "MyCoreMLRunner",
    platforms: [
        .macOS(.v10_15)
    ],
    products: [
        .executable(name: "my-core-ml-runner", targets: ["MyCoreMLRunner", "MyCoreMLModels"]),
        .library(name: "MyCoreMLModels", targets: ["MyCoreMLModels"]),
    ],
    targets: [
        .target(
            name: "MyCoreMLModels",
            dependencies: [],
            path: "Sources",
            // NOTE: .process() を使うとリソースの名前が変わるのか、bundle から取得できない
            resources: [.copy("./GPT2Model.mlmodelc")]),
        .executableTarget(
            name: "MyCoreMLRunner",
            dependencies: ["MyCoreMLModels"],
            path: "cmd"),
    ]
)
