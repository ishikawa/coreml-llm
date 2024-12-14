import CoreML
import Foundation
import GPT2Model
import Generation
import Models
import Tokenizers

class Generator: Generation {
    private let model: MLModel

    private let maxContextLength: Int

    init(model: MLModel, maxContextLength: Int) {
        self.model = model
        self.maxContextLength = maxContextLength
    }

    public func predictNextTokenScores(
        _ tokens: MLTensor,
        config: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2)  // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        let padLength = self.maxContextLength - tokenCount

        var inputIds = tokens
        if padLength > 0 {
            let padding = MLTensor(repeating: Int32(config.padTokenId ?? 0), shape: [1, padLength])
            inputIds = MLTensor(concatenating: [tokens, padding], alongAxis: -1)
        }

        var inputDictionary = ["inputIds": inputIds]

        var mask = [Int32](repeating: 1, count: tokenCount)

        // padLength 分だけ 0 を追加する
        if padLength > 0 {
            mask += [Int32](repeating: 0, count: padLength)
        }

        let attentionMask = MLTensor(shape: inputIds.shape, scalars: mask)
        inputDictionary["attentionMask"] = attentionMask

        let outputs = try! await model.prediction(from: inputDictionary)
        assert(outputs.keys.contains("logits"))

        let scores = outputs["logits"]!
        assert(scores.rank == 3)
        let tokenIndex = tokenCount - 1
        let nextTokenScores = scores[nil, tokenIndex, nil].expandingShape(at: 0)

        assert(nextTokenScores.rank == 3)
        assert(nextTokenScores.shape[0] == 1 && nextTokenScores.shape[1] == 1)
        return nextTokenScores
    }

    @discardableResult
    func generate(
        config: GenerationConfig,
        tokens: InputTokens,
        callback: PredictionTokensCallback?
    ) async -> GenerationOutput {
        return await generate(
            config: config,
            tokens: tokens,
            model: predictNextTokenScores,
            callback: callback)
    }
}

func run_command(prompt: String, maxLength: Int) async throws {
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
