import coremltools as ct
import torch
from transformers import AutoTokenizer

batch_size = 1
context_size = 2048
model_id = "meta-llama/Llama-3.2-1B-Instruct"

loaded_model = ct.models.MLModel("models/Llama-3.2-1B-Instruct.mlpackage")
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

# prompt
prompt = "What is generative AI?"
inputs = tokenizer(prompt, return_tensors="pt", max_length=2048, truncation=True)

input_ids = inputs["input_ids"]
attention_mask = inputs["attention_mask"]


# パディング関数を定義
def pad_to_context_size(tensor, context_size, pad_value=0):
    padded_tensor = torch.zeros((tensor.size(0), context_size), dtype=tensor.dtype)
    padded_tensor[:, : tensor.size(1)] = tensor
    return padded_tensor


# 生成ループの設定
max_generated_tokens = 50  # 最大生成トークン数
generated_tokens = []  # 生成されたトークンを格納

for i in range(max_generated_tokens):
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

    # input_idsとattention_maskを更新
    input_ids = torch.cat([input_ids, next_token_id.unsqueeze(0)], dim=1)
    attention_mask = torch.cat(
        [attention_mask, torch.ones((1, 1), dtype=torch.long)], dim=1
    )

    # コンテキストサイズを超えないようにする
    if input_ids.size(1) > context_size:
        input_ids = input_ids[:, -context_size:]
        attention_mask = attention_mask[:, -context_size:]

# 生成されたトークンをデコードしてテキストに変換
generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

print("Generated Text:", generated_text)
