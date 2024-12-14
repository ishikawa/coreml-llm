import CoreML
import Generation
import Models
import Tokenizers

// NOTE: swift-transformer (preview) では以下のようになっている
public class GPT2TextGenerationModel: TextGenerationModel {

    public static func load_model() throws -> Self {
        let configuration = MLModelConfiguration()
        // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
        let modelURL = Bundle.module.url(forResource: "GPT2Model", withExtension: "mlmodelc")!
        let model = try MLModel(contentsOf: modelURL, configuration: configuration)

        //let lm = LanguageModel(model: model)
        return Self(model: model)
    }

    public var defaultGenerationConfig: GenerationConfig
    public var modelName = "gpt2"

    public let minContextLength: Int
    public let maxContextLength: Int

    private var _tokenizer: Tokenizer?

    public var tokenizer: Tokenizer {
        get async throws {
            if let _tokenizer {
                return _tokenizer
            }
            _tokenizer = try await AutoTokenizer.from(pretrained: "gpt2")
            return _tokenizer!
        }
    }

    public var model: MLModel

    public required init(model: MLModel) {
        self.model = model
        (minContextLength, maxContextLength) = Self.getInputDescription(
            from: model, inputKey: "inputIds")
        self.defaultGenerationConfig = GenerationConfig(
            maxLength: maxContextLength,
            maxNewTokens: 128,
            doSample: true,
            topK: 50)
        self.defaultGenerationConfig.bosTokenId = 50256
        self.defaultGenerationConfig.eosTokenId = 50256
    }

    public func resetState() async {}

    public func predictNextTokenScores(
        _ tokens: MLTensor,
        config: GenerationConfig
    ) async -> MLTensor {
        assert(tokens.rank == 2)  // [batch, current sequence length]
        let tokenCount = tokens.shape[1]
        let padLength = maxContextLength - tokenCount

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

    static func getInputDescription(from model: MLModel, inputKey: String) -> (
        min: Int, max: Int
    ) {
        guard let inputDescription = model.modelDescription.inputDescriptionsByName[inputKey] else {
            fatalError("Cannot obtain input description")
        }

        guard let multiArrayConstraint = inputDescription.multiArrayConstraint else {
            fatalError("Cannot obtain shape information")
        }

        let shapeConstraint = multiArrayConstraint.shapeConstraint
        var minContextLength = 128
        var maxContextLength = 128

        switch shapeConstraint.type {
        case .enumerated:
            minContextLength = shapeConstraint.enumeratedShapes[0][1].intValue
            maxContextLength = minContextLength
        case .range:
            let sizeRangeForDimension = shapeConstraint.sizeRangeForDimension
            let lastAxis = sizeRangeForDimension.count - 1
            let range = sizeRangeForDimension[lastAxis] as? NSRange
            minContextLength = range?.location ?? 1
            maxContextLength = range?.length ?? 128
        case .unspecified:
            break
        @unknown default:
            break
        }

        return (minContextLength, maxContextLength)
    }
}
