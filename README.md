# coreml-llm

This is a personal project where I’m exploring how to run large language models (LLMs) on Mac and
iPhone using CoreML. I’m particularly interested in taking advantage of the Apple Neural Engine
(ANE) to make the models as fast and efficient as possible. I’m also experimenting with ways to
optimize LLMs for iPhone.

## Llama

```sh
$ poetry run python ./llama/export-model.py
$ poetry run python ./llama/predict.py
```

## GPT-2

### Export model

```sh
$ poetry run python ./gpt-2/baseline/export-model.py --context-size 1024 --minimum-deployment-target iOS18
$ make build
```

### Inference

Python

```
poetry run python ./gpt-2/baseline/python-generate.py --max-length=100 "What is generative AI?"
Generating text:
What is generative AI?

There are many applications that have different types of learning/learning in both real life languages (like languages for teaching or AI in an embedded data analysis application) and in real time applications (like machine learning or machine learning models), but generative AI in these applications would only be able to learn what is relevant to the question of what we learned using that same data.

Generative AI also has a "general purpose" approach where we ask if we could
[Prompt]  => 6 tokens, latency (TTFT): 1.06 ms
[Extend]  => 94 tokens, throughput: 27.67 tokens/s
```

Swift

```
swift run --package-path ./CoreMLRunner -- coreml-runner --max-length=106 "What is generative AI?"
Building for debugging...
[1/1] Write swift-version--58304C5D6DBC2206.txt
Build of product 'coreml-runner' complete! (0.14s)
What is generative AI? Read on to learn what that means, and how it could change what is happening in the future.
[Prompt]  => 6 tokens, latency (TTFT): 0.19 ms
[Extend]  => 20 tokens, throughput: 8.31 tokens/s
```

## Measure performance metrics

Python

```
poetry run python ./inference-metrics.py --warm 3 -n 5 -- poetry run python ./openai-gpt2/python-generate.py --max-length=106 "What is generative AI?"
...

Average Metrics:
[Prompt]  => 6 tokens, latency (TTFT): 0.29 ms
[Extend]  => 100 tokens, throughput: 44.93 tokens/s
```

Swift

```
poetry run python ./inference-metrics.py --warm 3 -n 5 -- swift run --package-path ./CoreMLRunner -- coreml-runner --max-length=106 "What is generative AI?"
...

Average Metrics:
[Prompt]  => 6 tokens, latency (TTFT): 0.15 ms
[Extend]  => 100 tokens, throughput: 9.03 tokens/s
```
