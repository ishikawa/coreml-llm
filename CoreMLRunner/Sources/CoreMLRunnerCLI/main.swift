import ArgumentParser
import CoreMLRunner

@main
struct CoreMLRunnerCLI: AsyncParsableCommand {
    @Argument(help: "The input prompt for text generation")
    var prompt: String

    @Option(name: .long, help: "Maximum length of the generated text")
    var maxLength: Int = 100

    mutating func run() async throws {
        try await CoreMLRunner.generate(
            prompt: prompt,
            maxLength: maxLength,
            doSample: true)
    }
}
