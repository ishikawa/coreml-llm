import CoreML

@available(macOS 13.0, iOS 16.0, tvOS 16.0, watchOS 9.0, visionOS 1.0, *)
public func llm_predict() {
  let configuration = MLModelConfiguration()
  // NOTE: Swift Package では Bundle.module でリソースにアクセスできる
  let modelURL = Bundle.module.url(forResource: "GPT2Model", withExtension:"mlmodelc")!
  let model = try? GPT2Model(contentsOf: modelURL, configuration: configuration)

  print("model: \(String(describing: model))")
}
