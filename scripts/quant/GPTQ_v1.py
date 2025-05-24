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



# ������������INT8��
int_bits = 8

# ģ��·��

model_id = "./models/Llama-3.2-3B-Instruct"
save_path = f"./Int{int_bits}_gptq_v1"


print("?? auto-gptq version:", auto_gptq.__version__)

# ============================== ����ģ����ִ��� ==============================

model = AutoModelForCausalLM.from_pretrained(model_id)
tokenizer = AutoTokenizer.from_pretrained(model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ============================== ������������ ==============================

quantize_config = BaseQuantizeConfig(
    bits=int_bits,
    group_size=32,
    desc_act=True,          # ���ü����Ż�
    sym=True,               # �Գ�����
    true_sequential=True    # ������������
)

# ============================== ׼��У׼���� ==============================

print("?? ׼��У׼���ݣ�GSM8K + ARC-Challenge")

# ---- GSM8K ----
gsm8k_dataset = load_dataset("gsm8k", "main", split="train")
gsm8k_samples = random.sample(list(gsm8k_dataset), min(100, len(gsm8k_dataset)))
gsm8k_texts = [f"Question: {item['question'].strip()}\nAnswer: " for item in gsm8k_samples]
print(f"?\t\t�� GSM8K �в����� {len(gsm8k_texts)} ��������������")

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
print(f"?\t\t�� ARC-Challenge �в����� {len(arc_texts)} ��У׼����")

# �ϲ�������
calibration_texts = gsm8k_texts + arc_texts
random.shuffle(calibration_texts)
print(f"?\t\t�ϲ��󹲲����� {len(calibration_texts)} ��У׼����")

# ����Ϊ����
calibration_data = tokenizer(
    calibration_texts,
    return_tensors="pt",
    padding=True,
    truncation=True,
    max_length=512
)

# ������ת�Ƶ� CPU�������Դ治�㣩
device = "cpu"
calibration_data = {k: v.to(device) for k, v in calibration_data.items()}

# ���� GPTQ �������������
examples = [
    {"input_ids": input_id, "attention_mask": attn_mask}
    for input_id, attn_mask in zip(calibration_data["input_ids"], calibration_data["attention_mask"])
]

# ============================== ִ������ ==============================

print(f"?? ��ʼִ�� INT{int_bits} ����...")
model = AutoGPTQForCausalLM.from_pretrained(
    model_id,
    quantize_config=quantize_config,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

model.quantize(examples=examples)

# ============================== ��������ģ�� ==============================

print(f"? ���� INT{int_bits} ����ģ�͵�: {save_path}")
model.save_quantized(save_path, use_safetensors=True)
tokenizer.save_pretrained(save_path)

print(f"?? INT{int_bits} ������ɣ�ģ���ѱ��浽: {save_path}")
