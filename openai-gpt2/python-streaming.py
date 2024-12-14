import click
from transformers import GPT2LMHeadModel, GPT2Tokenizer, TextStreamer

# GPT-2モデルとトークナイザーの読み込み
tokenizer = GPT2Tokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)
model = GPT2LMHeadModel.from_pretrained("gpt2")


def generate_streaming(prompt: str, max_length=50):
    encoded_inputs = tokenizer(prompt, return_tensors="pt")
    streamer = TextStreamer(
        tokenizer,
        skip_prompt=False,
        skip_special_tokens=True,
    )

    max_new_tokens = max_length - len(encoded_inputs["input_ids"][0])
    _ = model.generate(
        **encoded_inputs,
        streamer=streamer,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )


@click.command()
@click.argument("prompt", type=str)
@click.option(
    "--max_length",
    type=int,
    default=128,
    help="The maximum number of tokens to generate.",
)
def main(prompt, max_length):
    print("Generating text:")
    generate_streaming(prompt, max_length=max_length)


if __name__ == "__main__":
    main()
