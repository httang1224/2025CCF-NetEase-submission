#!/bin/bash

# ===============================
# LLaMA 模型评估脚本（只支持传入模型路径）
# ===============================
# 获取传入模型路径，若未传入则使用默认值
MODEL_PATH=${1:-"./models/Llama-3.2-3B-Instruct"}
OUTPUT_PATH="./outputs"
GPU_UTILIZATION=0.6
DTYPE="auto"
BATCH_SIZE="auto:1"
INPUT_LEN=4096
OUTPUT_LEN=100

echo "待评估模型路径: $MODEL_PATH"
echo "输出目录: $OUTPUT_PATH"
echo "==============================="

echo "开始推理延迟评估..."
python3 ./benchmarks/benchmark_latency.py \
  --model "$MODEL_PATH" \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN \
  --batch-size 1

echo "延迟评估完成！"

echo "开始 ARC-Challenge 精度评估..."
lm-eval --model vllm \
  --model_args pretrained="$MODEL_PATH",gpu_memory_utilization=$GPU_UTILIZATION,dtype=$DTYPE \
  --tasks arc_challenge \
  --batch_size $BATCH_SIZE \
  --output_path $OUTPUT_PATH

echo "ARC-Challenge 精度评估完成！"

echo "开始 GSM8K 精度评估..."
lm-eval --model vllm \
  --model_args pretrained="$MODEL_PATH",gpu_memory_utilization=$GPU_UTILIZATION,dtype=$DTYPE \
  --tasks gsm8k \
  --batch_size $BATCH_SIZE \
  --output_path $OUTPUT_PATH

echo "GSM8K 精度评估完成！"
echo "所有评估流程完成！结果保存在 $OUTPUT_PATH"