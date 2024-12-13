import CoreML
import Tokenizers

public func llm_predict() async throws{
  let configuration = MLModelConfiguration()
  // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
  let modelURL = Bundle.module.url(forResource: "GPT2Model", withExtension:"mlmodelc")!
  let model = try? GPT2Model(contentsOf: modelURL, configuration: configuration)

  print("model: \(String(describing: model))")


  let tokenizer = try await AutoTokenizer.from(pretrained: "gpt2")
  print("tokenizer: \(String(describing: tokenizer))")

  let inputIds = tokenizer("What is the meaning of life?")
  print("inputIds: \(inputIds)")
}
