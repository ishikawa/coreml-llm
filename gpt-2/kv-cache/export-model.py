from typing import Any, Optional, Sequence

import click
import coremltools as ct
import numpy as np
import torch
from transformers import Cache, GPT2Config
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

DEPLOYMENT_TARGETS = {
    "iOS13": ct.target.iOS13,
    "iOS14": ct.target.iOS14,
    "iOS15": ct.target.iOS15,
    "iOS16": ct.target.iOS16,
    "iOS17": ct.target.iOS17,
    "iOS18": ct.target.iOS18,
}


class BaselineGPT2LMHeadModel(GPT2LMHeadModel):
    """Baseline LlamaForCausalLM model without key/value caching."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        )
        return out.logits


# For KV cache with Stateful model in CoreML
# Borrowed from https://machinelearning.apple.com/research/core-ml-on-device-llama
class SliceUpdateKeyValueCache(Cache):
    """
    `SliceUpdateKeyValueCache`, that extends the `Cache` class.

    It essentially implements a simple update logic via the slicing operation, these op
    patterns are then detected by the Core ML-GPU compiler and allows it to perform
    in-place updates.
    """

    k: torch.Tensor
    v: torch.Tensor
    past_seen_tokens: int

    def __init__(
        self,
        *,
        shape: Sequence[int],
        dtype: torch.dtype = torch.float32,
    ) -> None:
        """Create key/value cache of shape:
        (#layers, batch_size, #kv_heads, context_size, head_dim)."""
        super().__init__()
        self.past_seen_tokens = 0
        self.k = torch.zeros(shape, dtype=dtype)
        self.v = torch.zeros(shape, dtype=dtype)

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Update key / value cache tensors for slice [begin, end).
        Return slice of key / value cache tensors from [0, end)."""
        if cache_kwargs is None:
            return key_states, value_states

        position: torch.Tensor = cache_kwargs.get("cache_position", None)
        assert position is not None, "cache_position required to update cache."

        begin, end = self.past_seen_tokens, self.past_seen_tokens + position.shape[-1]
        self.k[layer_idx, :, : key_states.shape[1], begin:end, :] = key_states
        self.v[layer_idx, :, : value_states.shape[1], begin:end, :] = value_states
        k_state = self.k[layer_idx, :, :, :end, :]
        v_state = self.v[layer_idx, :, :, :end, :]

        return k_state, v_state

    def get_seq_length(self, layer_idx: Optional[int] = 0) -> int:
        """Get the sequence length of the cache."""
        return self.past_seen_tokens

    # Support for backwards-compatible `past_key_value` indexing, e.g.
    # `past_key_value[0][0].shape[2]` to get the sequence length.
    def __getitem__(self, layer_idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        if layer_idx < len(self):
            # v[0].shape[2] == v[0].size(-2) == past_seen_tokens になるような v を作って返す
            k = torch.zeros(3, 1, self.past_seen_tokens, self.past_seen_tokens)
            v = torch.zeros(3, 1, self.past_seen_tokens, self.past_seen_tokens)

            return (k, v)
        else:
            raise KeyError(
                f"Cache only has {len(self)} layers, attempted to access layer with index {layer_idx}"
            )

    def __iter__(self):
        for layer_idx in range(len(self)):
            yield (self.k[layer_idx], self.v[layer_idx])

    def __len__(self):
        return self.k.size(0)


class KvCacheStateGPT2LMHeadModel(torch.nn.Module):
    """Model wrapper to swap cache implementation and register as buffers."""

    kv_cache_shape: tuple[int, ...]
    kv_cache: SliceUpdateKeyValueCache

    def __init__(self, *, batch_size: int = 1, context_size: int = 1024) -> None:
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained("gpt2").eval()
        config: GPT2Config = self.model.config

        self.kv_cache_shape: tuple[int, ...] = (
            config.n_layer,  # Number of hidden layers
            batch_size,
            config.n_head,  # Number of attention heads
            context_size,
            config.n_embd // config.n_head,  # Hidden size per head
        )

        # Register KV cache buffers to be recognized as Core ML states
        self.kv_cache = SliceUpdateKeyValueCache(shape=self.kv_cache_shape)
        self.register_buffer("keyCache", self.kv_cache.k)
        self.register_buffer("valueCache", self.kv_cache.v)

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        # Compute past seen tokens used for updating key/value cache slices
        self.kv_cache.past_seen_tokens = causal_mask.shape[-1] - input_ids.shape[-1]
        # print("past_seen_tokens", self.kv_cache.past_seen_tokens)
        # print("input_ids", input_ids)
        # print("causal_mask", causal_mask)

        assert causal_mask.dim() == 4, "Causal mask should have 4 dimensions."
        assert (
            causal_mask.max() == 0
        ), "Custom 4D attention mask should be passed in inverted form with max==0`"

        return self.model(
            input_ids=input_ids,
            attention_mask=causal_mask,
            past_key_values=self.kv_cache,
            use_cache=True,
        ).logits


@click.command()
@click.option(
    "--batch-size",
    default=1,
    help="Batch size for the model.",
    type=int,
)
@click.option(
    "--context-size",
    default=1024,
    help="Context window size for the model.",
    type=int,
)
@click.option(
    "--minimum-deployment-target",
    default="iOS18",
    help="Minimum deployment target (iOS13-iOS18).",
    type=click.Choice(list(DEPLOYMENT_TARGETS.keys())),
)
@click.option(
    "--output",
    default="models/gpt2-kv-cache.mlpackage",
    help="Output path for the Core ML model.",
    type=click.Path(),
)
def main(
    batch_size: int, context_size: int, minimum_deployment_target: str, output: str
):
    """Convert GPT-2 PyTorch model to Core ML format."""
    torch_model = KvCacheStateGPT2LMHeadModel(
        batch_size=batch_size, context_size=context_size
    ).eval()
    print("kv_cache_shape", torch_model.kv_cache_shape)

    example_inputs: tuple[torch.Tensor, ...] = (
        torch.zeros((1, 2), dtype=torch.long),
        torch.zeros((1, 1, 2, 5), dtype=torch.float32),
    )

    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        torch_model, example_inputs=example_inputs
    )
    # print("Traced model", traced_model)

    # Convert to Core ML
    query_size = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
    final_step = ct.RangeDim(lower_bound=1, upper_bound=context_size, default=1)
    inputs = [
        ct.TensorType(shape=(batch_size, query_size), dtype=np.int32, name="inputIds"),
        ct.TensorType(
            shape=(batch_size, 1, query_size, final_step),
            dtype=np.float16,
            name="causalMask",
        ),
    ]
    states = [
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=torch_model.kv_cache_shape, dtype=np.float16
            ),
            name="keyCache",
        ),
        ct.StateType(
            wrapped_type=ct.TensorType(
                shape=torch_model.kv_cache_shape, dtype=np.float16
            ),
            name="valueCache",
        ),
    ]
    outputs = [ct.TensorType(dtype=np.float16, name="logits")]

    mlmodel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        states=states,
        minimum_deployment_target=DEPLOYMENT_TARGETS[minimum_deployment_target],
        skip_model_load=False,
    )

    # set metadata
    mlmodel.input_description["inputIds"] = "Input token IDs."
    mlmodel.input_description["causalMask"] = "Causal mask for the model."
    mlmodel.output_description["logits"] = "Logits for next token prediction."

    mlmodel.author = "OpenAI"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Language Models are Unsupervised Multitask Learners"
    mlmodel.version = "2.0"

    mlmodel.save(output)


if __name__ == "__main__":
    main()
