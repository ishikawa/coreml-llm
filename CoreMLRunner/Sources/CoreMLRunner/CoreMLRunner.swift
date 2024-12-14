import CoreML
import Foundation
import Generation
import Models
import Tokenizers

public struct CoreMLRunner {
    public static func generate(prompt: String, maxLength: Int) async throws {
        let modelURL = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .appendingPathComponent("GPT2Model")
            .appendingPathExtension("mlmodelc")
        print("Loading model from \(modelURL)")

        // 簡潔さのために固定値にする。本来は、CoreML model のメタデータから取得する
        let maxContextLength = 1024

        let configuration = MLModelConfiguration()
        let model = try! MLModel(contentsOf: modelURL, configuration: configuration)

        let tokenizer = try await AutoTokenizer.from(pretrained: "gpt2")
        let inputIds = tokenizer.encode(text: prompt)

        var generationConfig = GenerationConfig(
            maxLength: maxLength,
            maxNewTokens: maxLength - inputIds.count,
            doSample: true)
        generationConfig.eosTokenId = tokenizer.eosTokenId
        generationConfig.bosTokenId = tokenizer.bosTokenId

        let generator = Generator(model: model, maxContextLength: maxContextLength)

        let output = await generator.generate(
            config: generationConfig,
            tokens: inputIds
        ) { tokens in
            let text = tokenizer.decode(tokens: tokens)

            print("Callback: \(text)")
        }

        let text = tokenizer.decode(tokens: output)
        print("\nOutput: \(text)")
    }
}
