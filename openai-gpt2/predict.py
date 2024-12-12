import time

import click
import coremltools as ct
import torch
from transformers import AutoTokenizer

batch_size = 1
context_size = 128
model_id = "gpt2"

loaded_model = ct.models.MLModel("models/gpt2.mlpackage")
tokenizer = AutoTokenizer.from_pretrained(
    model_id, use_fast=True, clean_up_tokenization_spaces=False
)


@click.command()
@click.argument("prompt", type=str)
@click.option("--max-length", default=128, help="Maximum length of generated text.")
def main(prompt, max_length):
    """
    Generate text using GPT-2 model."""
    # prompt
    inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)

    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # パディング関数を定義
    def pad_to_context_size(tensor, context_size, pad_value=0):
        padded_tensor = torch.zeros((tensor.size(0), context_size), dtype=tensor.dtype)
        padded_tensor[:, : tensor.size(1)] = tensor
        return padded_tensor

    # 生成ループの設定
    generated_tokens = []  # 生成されたトークンを格納
    start_time = time.time()
    ttft = None

    for i in range(max_length):
        # print(i, input_ids)

        # 現在の input_ids と attention_mask をコンテキストサイズにパディング
        input_ids_padded = pad_to_context_size(input_ids, context_size).to(torch.int32)
        attention_mask_padded = pad_to_context_size(attention_mask, context_size).to(
            torch.int32
        )

        # CoreMLモデルに入力
        logits = loaded_model.predict(
            {"inputIds": input_ids_padded, "attentionMask": attention_mask_padded}
        )

        # ロジットから次のトークンを取得 (貪欲法: 最大値のインデックスを選択)
        next_token_id = torch.tensor(
            [logits["logits"][0, input_ids.size(1) - 1, :].argmax()]
        )

        # <eos>トークンで終了チェック
        if next_token_id.item() == tokenizer.eos_token_id:
            break

        # 生成されたトークンを記録
        generated_tokens.append(next_token_id.item())

        # 最初のトークンが生成された時間を記録
        if len(generated_tokens) == 1:
            ttft = (time.time() - start_time) * 1000  # ミリ秒に変換

        # input_idsとattention_maskを更新
        input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1
        )

        # コンテキストサイズを超えないようにする
        if input_ids.size(1) > context_size:
            input_ids = input_ids[:, -context_size:]
            attention_mask = attention_mask[:, -context_size:]

    end_time = time.time()
    total_time = end_time - start_time
    tokens_per_second = len(generated_tokens) / total_time

    # 生成されたトークンをデコードしてテキストに変換
    generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

    print("Generated Text:", generated_text)

    print("\nPerformance Metrics:")
    print(f"Time to First Token (TTFT): {ttft:.2f} ms")
    print(f"Tokens Per Second (TPS): {tokens_per_second:.2f}")


if __name__ == "__main__":
    main()
