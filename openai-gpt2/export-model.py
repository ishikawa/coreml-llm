import coremltools as ct
import numpy as np
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

model_id = "gpt2"

batch_size = 1
context_size = 128
input_shape = (batch_size, context_size)


# print(model.__class__.__name__)


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


torch_model = BaselineGPT2LMHeadModel.from_pretrained(model_id).eval()

# trace the PyTorch model
example_inputs: tuple[torch.Tensor, torch.Tensor] = (
    torch.zeros(input_shape, dtype=torch.long),
    torch.zeros(input_shape, dtype=torch.long),
)
traced_model: torch.jit.ScriptModule = torch.jit.trace(
    torch_model,
    example_inputs=example_inputs,
)

# GPT2LMHeadModel
# input_ids torch.Size([1, 11])
# attention_mask torch.Size([1, 11])
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
    minimum_deployment_target=ct.target.macOS13,
    skip_model_load=True,
)

mlmodel.save("models/gpt2.mlpackage")
