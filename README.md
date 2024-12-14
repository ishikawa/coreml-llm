# coreml-llm

This is a personal project where I’m exploring how to run large language models (LLMs) on Mac and
iPhone using CoreML. I’m particularly interested in taking advantage of the Apple Neural Engine
(ANE) to make the models as fast and efficient as possible. I’m also experimenting with ways to
optimize LLMs for iPhone.

## Models

### Llama

```sh
$ poetry run python ./llama/export-model.py
$ poetry run python ./llama/predict.py
```

### GPT-2

Python

```sh
poetry run python ./openai-gpt2/python-generate.py --max-length=100 "What is generative AI?"
Generating text:
What is generative AI?

There are many applications that have different types of learning/learning in both real life languages (like languages for teaching or AI in an embedded data analysis application) and in real time applications (like machine learning or machine learning models), but generative AI in these applications would only be able to learn what is relevant to the question of what we learned using that same data.

Generative AI also has a "general purpose" approach where we ask if we could
[Prompt]  => 6 tokens, latency (TTFT): 1.06 ms
[Extend]  => 94 tokens, throughput: 27.67 tokens/s
```
