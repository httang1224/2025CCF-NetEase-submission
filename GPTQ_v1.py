# -*- coding: utf-8 -*-
# Author  : haitong
# Created : 2025-05-24
# File    : GPTQ_v1.py



import os
import random
import torch
import auto_gptq
from transformers import AutoTokenizer, AutoModelForCausalLM
from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
from datasets import load_dataset



# 量化比特数（INT8）
int_bits = 8

# 模型路径

model_id = "./models/Llama-3.2-3B-Instruct"
save_path = f"./Int{int_bits}_gptq_v1"


print("?? auto-gptq version:", auto_gptq.__version__)

# ============================== 加载模型与分词器 ==============================

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================== 配置量化参数 ==============================

quantize_config = BaseQuantizeConfig(
    bits=int_bits,
    group_size=32,
    desc_act=True,          # 启用激活优化
    sym=True,               # 对称量化
    true_sequential=True    # 启用逐行误差补偿
)

# ============================== 准备校准数据 ==============================

print("?? 准备校准数据：GSM8K + ARC-Challenge")

# ---- GSM8K ----
gsm8k_dataset = load_dataset("gsm8k", "main", split="train")
gsm8k_samples = random.sample(list(gsm8k_dataset), min(100, len(gsm8k_dataset)))
gsm8k_texts = [f"Question: {item['question'].strip()}\nAnswer: " for item in gsm8k_samples]
print(f"?\t\t从 GSM8K 中采样了 {len(gsm8k_texts)} 条量化输入样本")

# ---- ARC-Challenge ----
arc_train = load_dataset("ai2_arc", "ARC-Challenge", split="train")
arc_samples = random.sample(list(arc_train), min(100, len(arc_train)))
arc_all = arc_samples + arc_samples

arc_texts = []
for item in arc_all:
    q = f"Question: {item['question'].strip()}\n"
    for idx, choice in enumerate(item["choices"]["text"]):
        q += f"({chr(65 + idx)}) {choice.strip()}\n"
    q += "Answer: "
    arc_texts.append(q)
print(f"?\t\t从 ARC-Challenge 中采样了 {len(arc_texts)} 条校准样本")

# 合并并打乱
calibration_texts = gsm8k_texts + arc_texts
random.shuffle(calibration_texts)
print(f"?\t\t合并后共采样了 {len(calibration_texts)} 条校准样本")

# 编码为张量
calibration_data = tokenizer(
    calibration_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# 将数据转移到 CPU（避免显存不足）
device = "cpu"
calibration_data = {k: v.to(device) for k, v in calibration_data.items()}

# 构造 GPTQ 所需的输入样例
examples = [
    {"input_ids": input_id, "attention_mask": attn_mask}
    for input_id, attn_mask in zip(calibration_data["input_ids"], calibration_data["attention_mask"])
]

# ============================== 执行量化 ==============================

print(f"?? 开始执行 INT{int_bits} 量化...")
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.quantize(examples=examples)

# ============================== 保存量化模型 ==============================

print(f"? 保存 INT{int_bits} 量化模型到: {save_path}")
model.save_quantized(save_path, use_safetensors=True)
tokenizer.save_pretrained(save_path)

print(f"?? INT{int_bits} 量化完成！模型已保存到: {save_path}")
