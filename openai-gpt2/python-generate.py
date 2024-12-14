import time
from typing import Optional

import click
from transformers import AutoTokenizer, GPT2LMHeadModel, GPT2Tokenizer, TextStreamer


class PerformanceMetricsStreamer(TextStreamer):
    start_time: float
    end_time: Optional[float]
    first_token_time: Optional[float]
    n_tokens_generated: int

    def __init__(
        self, tokenizer: AutoTokenizer, skip_prompt: bool = False, **decode_kwargs
    ):
        super().__init__(tokenizer, skip_prompt, **decode_kwargs)
        self.start_time = time.time()
        self.end_time = None
        self.first_token_time = None
        self.n_tokens_generated = 0

    def put(self, value):
        # プロンプトは無視
        if not self.next_tokens_are_prompt:
            if not self.first_token_time:
                self.first_token_time = time.time() - self.start_time

            # トークン数をカウント
            if len(value.shape) > 1:
                self.n_tokens_generated += value.shape[1]
            else:
                self.n_tokens_generated += value.shape[0]

        super().put(value)

        # プロンプト出力は完了しているはずなので、フラグを外す
        self.next_tokens_are_prompt = False

    def end(self):
        super().end()
        self.end_time = time.time()


@click.command()
@click.argument("prompt", type=str)
@click.option(
    "--max-length",
    type=int,
    default=100,
    help="The maximum number of tokens to generate.",
)
def main(prompt, max_length):
    tokenizer = GPT2Tokenizer.from_pretrained(
        "gpt2", clean_up_tokenization_spaces=False
    )
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    encoded_inputs = tokenizer(prompt, return_tensors="pt")
    streamer = PerformanceMetricsStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
    )

    n_prompt_tokens = len(encoded_inputs["input_ids"][0])
    max_new_tokens = max_length - n_prompt_tokens

    print("Generating text:")
    _ = model.generate(
        **encoded_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Example:
    # [Prompt]  => 7 tokens, latency (TTFT): 5374.15 ms
    # [Extend]  => 100 tokens, throughput: 0.19 tokens/s
    print(
        f"[Prompt]  => {n_prompt_tokens} tokens, latency (TTFT): {streamer.first_token_time:.2f} ms"
    )
    print(
        f"[Extend]  => {streamer.n_tokens_generated} tokens, throughput: {streamer.n_tokens_generated / (streamer.end_time - streamer.start_time):.2f} tokens/s"
    )


if __name__ == "__main__":
    main()
