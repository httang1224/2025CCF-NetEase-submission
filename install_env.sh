#!/bin/bash

echo "?? ���ڴ��� Conda ������llm_compress"

# ���� Conda ����
conda create -n llmcompress python=3.9 -y

# ��ʼ�� Conda ������ȷ���ű�����ʹ�� conda activate��
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmcompress

echo "? Conda �����Ѽ����ʼ��װ����..."

# ��װ PyTorch��CUDA 12.1���汾
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# ��װ���� Python ����
pip install auto-gptq==0.7.1
pip install datasets==2.17.0
pip install vllm==0.7.1
pip install lm-eval==0.4.8
pip install huggingface_hub

pip install autoawq
pip install nvitop
pip install ipykernel
pip install matplotlib
pip install seaborn

echo ""
echo "����������װ��ɣ�"
