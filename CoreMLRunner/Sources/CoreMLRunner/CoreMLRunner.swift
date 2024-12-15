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

        let streamer = PerformanceMetricsStreamer(tokenizer: tokenizer)
        let generator = Generator(model: model, maxContextLength: maxContextLength)
        generator.streamer = streamer

        _ = await generator.generate(
            config: generationConfig,
            tokens: inputIds
        )

        // Example output:
        // [Prompt]  => 7 tokens, latency (TTFT): 5374.15 ms
        // [Extend]  => 100 tokens, throughput: 0.19 tokens/s

        //print("numTokensGenerated: \(streamer.numTokensGenerated)")
        //print("numPromptTokens: \(streamer.numPromptTokens)")

        let numPromptTokens = inputIds.count
        let n = streamer.numTokensGenerated - numPromptTokens

        print(
            "[Prompt]  => \(numPromptTokens) tokens, latency (TTFT): \(String(format: "%.2f", streamer.firstTokenTime!)) ms"
        )
        print(
            "[Extend]  => \(n) tokens, throughput: \(String(format: "%.2f", Double(n) / (streamer.endTime!.timeIntervalSince(streamer.startTime)))) tokens/s"
        )
    }

    static func load_model() throws -> MLModel {
        let configuration = MLModelConfiguration()
        // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
        let modelURL = Bundle.module.url(forResource: "gpt2-baseline", withExtension: "mlmodelc")!
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
