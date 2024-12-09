# type: ignore
import torch
from transformers.models.llama.modeling_llama import LlamaForCausalLM


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


model_id: str = "meta-llama/Llama-3.1-8B-Instruct"
torch_model = BaselineLlamaForCausalLM.from_pretrained(model_id).eval()
