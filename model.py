# type: ignore
import coremltools as ct
import numpy as np
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM

model_id = "meta-llama/Llama-3.1-8B-Instruct"

batch_size = 1
context_size = 2048
input_shape = (batch_size, context_size)


class BaselineLlamaForCausalLM(LlamaForCausalLM):
    """Baseline LlamaForCausalLM model without key/value caching."""

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: torch.LongTensor,
    ) -> torch.Tensor:
        out = super().forward(
            input_ids,
            attention_mask,
            use_cache=False,
        )
        return out.logits


torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()

# trace the PyTorch model
example_inputs: tuple[torch.Tensor] = (
    torch.zeros(input_shape, dtype=torch.int32),
    torch.zeros(input_shape, dtype=torch.int32),
)
traced_model: torch.jit.ScriptModule = torch.jit.trace(
    torch_model,
    example_inputs=example_inputs,
)

# convert to Core ML format
inputs: list[ct.TensorType] = [
    ct.TensorType(shape=input_shape, dtype=np.int32, name="inputIds"),
    ct.TensorType(shape=input_shape, dtype=np.int32, name="attentionMask"),
]

outputs: list[ct.TensorType] = [ct.TensorType(dtype=np.float16, name="logits")]
mlmodel: ct.models.MLModel = ct.convert(
    traced_model,
    inputs=inputs,
    outputs=outputs,
    minimum_deployment_target=ct.target.macOS13,
    skip_model_load=True,
)
