import CoreML
import Generation
import Models
import Tokenizers

public class GPT2TextGenerationModel: TextGenerationModel {

    public var defaultGenerationConfig: GenerationConfig

    public var modelName = "gpt2"

    public var inputIdsShape: [Int]
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
        (inputIdsShape, minContextLength, maxContextLength) = Self.getInputDescription(
            from: model, inputKey: "inputIds")
        self.defaultGenerationConfig = GenerationConfig(
            maxLength: 128,
            maxNewTokens: 128,
            doSample: true)
    }

    // TODO: Use MLTensor
    // MLShapedArrayProtocol is either a MLShapedArray or a MLShapedArraySlice
    public func predictNextTokenScores(_ tokens: InputTokens, config: GenerationConfig)
        -> any MLShapedArrayProtocol
    {
        // TODO: exceptions

        // Maybe pad or truncate
        let maxTokens = min(tokens.count, maxContextLength)
        let padLength = maxTokens >= minContextLength ? 0 : minContextLength - maxTokens
        let inputTokens =
            Array(tokens[0..<maxTokens])
            + Array(repeating: config.padTokenId ?? 0, count: padLength)

        let inputIds = MLShapedArray<Int32>(
            scalars: inputTokens.map { Int32($0) }, shape: inputIdsShape)
        var inputDictionary = ["inputIds": MLFeatureValue(shapedArray: inputIds)]

        // Attention mask
        let mask = Array(repeating: 1, count: maxTokens) + Array(repeating: 0, count: padLength)
        let attentionMask = MLShapedArray<Int32>(
            scalars: mask.map { Int32($0) }, shape: inputIdsShape)
        inputDictionary["attentionMask"] = MLFeatureValue(shapedArray: attentionMask)

        let input = try! MLDictionaryFeatureProvider(dictionary: inputDictionary)

        let output = try! model.prediction(from: input)

        // TODO: maybe try to support models with "token_scores" too (after the softmax)
        assert(output.featureNames.first! == "logits")

        let scores = output.featureValue(for: output.featureNames.first!)!.shapedArrayValue(
            of: Float.self)!
        let nextTokenScores = scores[0, maxTokens - 1]

        return nextTokenScores
    }

    static func getInputDescription(from model: MLModel, inputKey: String) -> (
        shape: [Int], min: Int, max: Int
    ) {
        guard let inputDescription = model.modelDescription.inputDescriptionsByName[inputKey] else {
            fatalError("Cannot obtain input description")
        }

        guard let multiArrayConstraint = inputDescription.multiArrayConstraint else {
            fatalError("Cannot obtain shape information")
        }

        let shapeConstraint = multiArrayConstraint.shapeConstraint

        let inputIdsShape = multiArrayConstraint.shape.map { $0.intValue }

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

        return (inputIdsShape, minContextLength, maxContextLength)
    }
}

public func llm_predict() async throws -> String {
    let configuration = MLModelConfiguration()
    // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
    let modelURL = Bundle.module.url(forResource: "GPT2Model", withExtension: "mlmodelc")!
    let model = try MLModel(contentsOf: modelURL, configuration: configuration)
    let lm = GPT2TextGenerationModel(model: model)

    return try await lm.generate(config: lm.defaultGenerationConfig, prompt: "Hello, my name is")
}
