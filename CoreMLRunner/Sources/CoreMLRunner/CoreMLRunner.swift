import CoreML
import Foundation
import Generation
import Models
import Tokenizers

public struct CoreMLRunner {
    public static func generate(prompt: String, maxLength: Int, doSample: Bool = false) async throws
    {
        let model = try load_model()
        let (_, maxContextLength) = getInputDescription(from: model, inputKey: "inputIds")

        let tokenizer = try await AutoTokenizer.from(pretrained: "gpt2")
        let inputIds = tokenizer.encode(text: prompt)

        var generationConfig = GenerationConfig(
            maxLength: maxLength,
            maxNewTokens: maxLength - inputIds.count,
            doSample: doSample)
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
