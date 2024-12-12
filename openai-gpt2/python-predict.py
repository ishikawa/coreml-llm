import time

import click
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


@click.command()
@click.argument("prompt", type=str)
@click.option("--max-length", default=128, help="Maximum length of generated text.")
@click.option("--model-name", default="gpt2", help="Model name or path for GPT-2.")
def main(prompt, max_length, model_name):
    # Load model and tokenizer
    print("Loading model and tokenizer...")
    model = GPT2LMHeadModel.from_pretrained(model_name)
    tokenizer = GPT2Tokenizer.from_pretrained(
        model_name, clean_up_tokenization_spaces=False
    )

    model.generation_config.pad_token_id = (
        tokenizer.pad_token_id or tokenizer.eos_token_id
    )
    model.eval()

    # Prepare input
    encoded_inputs = tokenizer(prompt, return_tensors="pt")

    # Generate text and measure performance
    print("Generating text...")

    # Measure time to first token (TTFT)
    start_time = time.time()
    with torch.no_grad():
        model.generate(
            **encoded_inputs,
            max_new_tokens=1,  # Generate only the first token
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    first_token_time = time.time() - start_time

    # Generate full text
    start_time = time.time()
    with torch.no_grad():
        outputs = model.generate(
            **encoded_inputs,
            max_length=max_length,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    # Decode the output
    generated_text = tokenizer.decode(
        outputs[0],
        skip_special_tokens=True,
    )

    # Measure throughput (TPS)
    total_tokens = len(outputs[0]) - len(
        encoded_inputs.input_ids[0]
    )  # Exclude prompt tokens
    generate_time = time.time() - start_time
    tps = total_tokens / generate_time if total_tokens > 0 else 0

    # Print results
    print("\nGenerated Text:")
    print(generated_text)

    print("\nPerformance Metrics:")
    print(f"Time to First Token (TTFT): {first_token_time * 1000:.2f} ms")
    print(f"Tokens Per Second (TPS): {tps:.2f}")


if __name__ == "__main__":
    main()
