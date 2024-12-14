import CoreML
import Foundation
import MyCoreMLModels

guard CommandLine.arguments.count > 1 else {
    print("Usage: command <input text>")
    exit(1)
}

let prompt = CommandLine.arguments[1]
let output = try! await llm_predict(prompt: prompt)

print("Output: \(output)")
