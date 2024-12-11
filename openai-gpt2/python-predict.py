import click
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2", clean_up_tokenization_spaces=False)


@click.command()
@click.argument("prompt", default="What is generative AI?", required=False)
def main(prompt):
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    gen_tokens = model.generate(
        input_ids,
        do_sample=True,
        temperature=0.01,
        max_length=100,
    )

    gen_text = tokenizer.batch_decode(gen_tokens, clean_up_tokenization_spaces=False)[0]

    print("Prompt:", prompt)
    print("Generated text:", gen_text)


if __name__ == "__main__":
    main()
