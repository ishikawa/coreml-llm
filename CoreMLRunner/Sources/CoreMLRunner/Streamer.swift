import Foundation
import Generation
import Tokenizers

/// Based on HF's transformer library
/// https://huggingface.co/docs/transformers/internal/generation_utils#transformers.TextStreamer
protocol BaseStreamer {
    /// Function that is called by `.generate()` to push current iteration tokens
    func put(_ value: GenerationOutput)

    /// Function that is called by `.generate()` to signal the end of generation
    func end()
}

class TextStreamer: BaseStreamer {
    private let tokenizer: Tokenizer

    // Variables used in streaming process
    private var tokenCache: [Int] = []
    private var printLen: Int = 0

    private(set) public var numTokensGenerated: Int = 0

    init(tokenizer: Tokenizer) {
        self.tokenizer = tokenizer
    }

    func put(_ value: GenerationOutput) {
        // Add new tokens to cache and decode
        let numNewTokens = value.count - numTokensGenerated

        // value の末尾 numNewTokens だけを tokenCache に追加する
        if numNewTokens > 0 {
            tokenCache.append(contentsOf: value.suffix(numNewTokens))
            numTokensGenerated = value.count
        }

        let text = tokenizer.decode(tokens: tokenCache)

        var printableText: String

        if text.hasSuffix("\n") {
            printableText = String(text.dropFirst(printLen))
            tokenCache.removeAll()
            printLen = 0
        } else if !text.isEmpty && isChineseChar(text.last?.unicodeScalars.first?.value ?? 0) {
            printableText = String(text.dropFirst(printLen))
            printLen += printableText.count
        } else {
            let lastSpaceIndex = text.lastIndex(of: " ") ?? text.startIndex
            let substringEnd = text.index(after: lastSpaceIndex)
            printableText = String(
                text[text.index(text.startIndex, offsetBy: printLen)..<substringEnd])
            printLen += printableText.count
        }

        onFinalizedText(printableText)
    }

    func end() {
        var printableText = ""
        if !tokenCache.isEmpty {
            let text = tokenizer.decode(tokens: tokenCache)

            printableText = String(text.dropFirst(printLen))
            tokenCache.removeAll()
            printLen = 0
        }

        onFinalizedText(printableText, streamEnd: true)
    }

    private func onFinalizedText(_ text: String, streamEnd: Bool = false) {
        print(text, terminator: streamEnd ? "\n" : "")
        fflush(stdout)
    }

    private func isChineseChar(_ cp: UInt32) -> Bool {
        return (cp >= 0x4E00 && cp <= 0x9FFF) || (cp >= 0x3400 && cp <= 0x4DBF)
            || (cp >= 0x20000 && cp <= 0x2A6DF) || (cp >= 0x2A700 && cp <= 0x2B73F)
            || (cp >= 0x2B740 && cp <= 0x2B81F) || (cp >= 0x2B820 && cp <= 0x2CEAF)
            || (cp >= 0xF900 && cp <= 0xFAFF) || (cp >= 0x2F800 && cp <= 0x2FA1F)
    }
}

final class PerformanceMetricsStreamer: TextStreamer {
    public let startTime: Date
    private(set) public var endTime: Date?
    private(set) public var firstTokenTime: TimeInterval?

    override init(tokenizer: Tokenizer) {
        self.startTime = Date()
        super.init(tokenizer: tokenizer)
    }

    override func put(_ value: GenerationOutput) {
        // swift-transformers の実装では、最初のトークン生成後に put が呼ばれる
        if firstTokenTime == nil {
            firstTokenTime = Date().timeIntervalSince(startTime)
        }

        super.put(value)
    }

    override func end() {
        super.end()
        endTime = Date()
    }
}
