import ArgumentParser

@main
struct GPT2Runner: AsyncParsableCommand {
    @Argument(help: "The input prompt for text generation")
    var prompt: String

    @Option(name: .long, help: "Maximum length of the generated text")
    var maxLength: Int = 100

    mutating func run() async throws {
        try await run_command(prompt: prompt, maxLength: maxLength)
    }
}
