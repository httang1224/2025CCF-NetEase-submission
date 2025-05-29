#!/bin/bash

# ===============================
# LLaMA ģ�������ű���ֻ֧�ִ���ģ��·����
# ===============================
# ��ȡ����ģ��·������δ������ʹ��Ĭ��ֵ
MODEL_PATH=${1:-"./models/Llama-3.2-3B-Instruct"}
OUTPUT_PATH="./outputs"
GPU_UTILIZATION=0.6
DTYPE="auto"
BATCH_SIZE="auto:1"
INPUT_LEN=4096
OUTPUT_LEN=100

echo "������ģ��·��: $MODEL_PATH"
echo "���Ŀ¼: $OUTPUT_PATH"
echo "==============================="

echo "��ʼ�����ӳ�����..."
python3 ./benchmarks/benchmark_latency.py \
  --model "$MODEL_PATH" \
  --input-len $INPUT_LEN \
  --output-len $OUTPUT_LEN \
  --batch-size 1

echo "�ӳ�������ɣ�"

echo "��ʼ ARC-Challenge ��������..."
lm-eval --model vllm \
  --model_args pretrained="$MODEL_PATH",gpu_memory_utilization=$GPU_UTILIZATION,dtype=$DTYPE \
  --tasks arc_challenge \
  --batch_size $BATCH_SIZE \
  --output_path $OUTPUT_PATH

echo "ARC-Challenge ����������ɣ�"

echo "��ʼ GSM8K ��������..."
lm-eval --model vllm \
  --model_args pretrained="$MODEL_PATH",gpu_memory_utilization=$GPU_UTILIZATION,dtype=$DTYPE \
  --tasks gsm8k \
  --batch_size $BATCH_SIZE \
  --output_path $OUTPUT_PATH

echo "GSM8K ����������ɣ�"
echo "��������������ɣ���������� $OUTPUT_PATH"