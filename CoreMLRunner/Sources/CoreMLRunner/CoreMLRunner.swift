import CoreML
import Foundation
import Generation
import Models
import Tokenizers

public struct CoreMLRunner {
    public static func generate(prompt: String, maxLength: Int) async throws {
        // 簡潔さのために固定値にする。本来は、CoreML model のメタデータから取得する
        let maxContextLength = 1024

        let model = try load_model()

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

    static func load_model() throws -> MLModel {
        let configuration = MLModelConfiguration()
        // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
        let modelURL = Bundle.module.url(forResource: "GPT2Model", withExtension: "mlmodelc")!
        return try MLModel(contentsOf: modelURL, configuration: configuration)
    }

}
