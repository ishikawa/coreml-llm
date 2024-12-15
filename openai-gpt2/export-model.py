import click
import coremltools as ct
import numpy as np
import torch
from transformers.models.gpt2.modeling_gpt2 import GPT2LMHeadModel

model_id = "gpt2"

batch_size = 1
context_size = 1024
input_shape = (batch_size, context_size)

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
    model_id = "gpt2"
    batch_size = 1
    input_shape = (batch_size, context_size)

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
