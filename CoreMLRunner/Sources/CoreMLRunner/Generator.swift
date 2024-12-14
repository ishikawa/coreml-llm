import CoreML
import Foundation
import Generation
import Models
import Tokenizers

class Generator: Generation {
    private let model: MLModel

    private let maxContextLength: Int

    public var streamer: BaseStreamer?

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
    func generate(config: GenerationConfig, tokens: InputTokens) async -> GenerationOutput {
        let output = await generate(
            config: config,
            tokens: tokens,
            model: predictNextTokenScores
        ) { tokens in
            self.streamer?.put(tokens)
        }

        streamer?.end()
        return output
    }
}
