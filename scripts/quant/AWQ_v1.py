# -*- encoding: utf-8 -*-
# Author  : haitong
# Time    : 2025-05-29 22:59
# File    : AWQ_v1.py
# Software: PyCharm

import random
from datasets import load_dataset
from transformers import AutoTokenizer
from awq import AutoAWQForCausalLM

# ============================== 基本配置 ==============================

# ⚠️ AWQ 当前仅支持 4-bit 权重量化
int_bits = 4
assert int_bits == 4, "❌ AWQ 目前仅支持 int_bits = 4"

# 模型路径
model_id = "./models/Llama-3.2-3B-Instruct"
save_path = f"./Int{int_bits}_awq_v1"

# ============================== 加载模型与分词器 ==============================

print("✅ 加载模型与分词器中...")
model = AutoAWQForCausalLM.from_pretrained(
    model_id,
    low_cpu_mem_usage=True,
    use_cache=False
)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================== 准备校准数据集 ==============================

print("✅ 准备校准数据集：GSM8K + ARC-Challenge")

# ---- GSM8K ----
gsm_dataset = load_dataset("gsm8k", "main", split="train")
gsm_samples = random.sample(list(gsm_dataset), 100)
gsm_texts = [f"Question: {item['question'].strip()}\nAnswer: " for item in gsm_samples]
print(f"  ✅ GSM8K 样本数: {len(gsm_texts)}")

# ---- ARC-Challenge ----
arc_train = load_dataset("ai2_arc", "ARC-Challenge", split="train")
arc_val = load_dataset("ai2_arc", "ARC-Challenge", split="validation")
arc_samples = random.sample(list(arc_train), 80) + random.sample(list(arc_val), 20)

arc_texts = []
for item in arc_samples:
    q = f"Question: {item['question'].strip()}\n"
    for idx, choice in enumerate(item["choices"]["text"]):
        q += f"({chr(65 + idx)}) {choice.strip()}\n"
    q += "Answer: "
    arc_texts.append(q)
print(f"  ✅ ARC-Challenge 样本数: {len(arc_texts)}")

# 合并 + 打乱
calibration_texts = gsm_texts + arc_texts
random.shuffle(calibration_texts)
print(f"  ✅ 合并后共计: {len(calibration_texts)} 条校准样本")

# ============================== 设置量化配置 ==============================

print("✅ 设置 AWQ 量化配置")
quant_config = {
    "zero_point": True,
    "q_group_size": 64,
    "w_bit": int_bits,
    "version": "GEMM"  # 也可使用 TURBO，但需自行编译
}

# ============================== 执行 AWQ 量化 ==============================

print(f"✅ 开始执行 INT{int_bits} 量化...")
model.quantize(
    tokenizer=tokenizer,
    quant_config=quant_config,
    calib_data=calibration_texts,
    n_parallel_calib_samples=64
)

# ============================== 保存量化模型 ==============================

print(f"✅ 保存 INT{int_bits} 量化模型至: {save_path}")
model.save_quantized(save_path, safetensors=True)
tokenizer.save_pretrained(save_path)

print(f"🎉 INT{int_bits} 量化完成，模型保存成功！")
