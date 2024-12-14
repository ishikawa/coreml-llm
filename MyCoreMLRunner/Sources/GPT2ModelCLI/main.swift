import CoreML
import Foundation
import GPT2Model

guard CommandLine.arguments.count > 1 else {
    print("Usage: command <input text>")
    exit(1)
}

let prompt = CommandLine.arguments[1]
let lm = try! GPT2TextGenerationModel.load_model()
let output = try! await lm.generate(config: lm.defaultGenerationConfig, prompt: prompt)

print("Output: \(output)")
