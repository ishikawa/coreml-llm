from typing import Any, Optional, Sequence

import click
import coremltools as ct
import numpy as np
import torch
from transformers import Cache
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


@click.command()
@click.option(
    "--context-size",
    default=1024,
    help="Context window size for the model.",
    type=int,
)
@click.option(
    "--minimum-deployment-target",
    default="iOS16",
    help="Minimum deployment target (iOS13-iOS18).",
    type=click.Choice(list(DEPLOYMENT_TARGETS.keys())),
)
@click.option(
    "--output",
    default="models/GPT2Model.mlpackage",
    help="Output path for the Core ML model.",
    type=click.Path(),
)
def main(context_size: int, minimum_deployment_target: str, output: str):
    """Convert GPT-2 PyTorch model to Core ML format."""
    batch_size = 1
    input_shape = (batch_size, context_size)

    torch_model = BaselineGPT2LMHeadModel.from_pretrained("gpt2").eval()

    # trace the PyTorch model
    example_inputs: tuple[torch.Tensor, torch.Tensor] = (
        torch.zeros(input_shape, dtype=torch.long),
        torch.zeros(input_shape, dtype=torch.long),
    )
    traced_model: torch.jit.ScriptModule = torch.jit.trace(
        torch_model,
        example_inputs=example_inputs,
    )

    # convert to Core ML format
    inputs: list[ct.TensorType] = [
        ct.TensorType(shape=input_shape, dtype=np.long, name="inputIds"),
        ct.TensorType(shape=input_shape, dtype=np.long, name="attentionMask"),
    ]

    outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
    mlmodel: ct.models.MLModel = ct.convert(
        traced_model,
        inputs=inputs,
        outputs=outputs,
        minimum_deployment_target=DEPLOYMENT_TARGETS[minimum_deployment_target],
        skip_model_load=True,
    )

    # set metadata
    mlmodel.input_description["inputIds"] = "Input token IDs."
    mlmodel.input_description["attentionMask"] = "Attention mask."
    mlmodel.output_description["logits"] = "Logits for next token prediction."

    mlmodel.author = "OpenAI"
    mlmodel.license = "MIT"
    mlmodel.short_description = "Language Models are Unsupervised Multitask Learners"
    mlmodel.version = "2.0"

    mlmodel.save(output)


if __name__ == "__main__":
    main()
