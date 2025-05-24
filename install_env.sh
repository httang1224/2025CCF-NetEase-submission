#!/bin/bash

echo "?? ���ڴ��� Conda ������llm_compress"

# Conda ��ʼ��
source "$(conda info --base)/etc/profile.d/conda.sh"

# �жϻ����Ƿ��Ѵ���
if conda env list | grep -q "llm_compress"; then
  echo "?? ���� llm_compress �Ѵ��ڣ���������..."
else
  conda create -n llm_compress python=3.9 -y
fi

conda activate llm_compress
echo "? Conda �����Ѽ����ʼ��װ����..."

# ��װ PyTorch��ע����� vllm��
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# ��װ��Ҫ����
pip install auto-gptq==0.7.1 datasets==2.17.0 vllm==0.7.1 \
lm-eval==0.4.8 huggingface_hub autoawq nvitop ipykernel matplotlib seaborn

echo ""
echo "?? ����������װ��ɣ��������� 'python -c \"from vllm import LLM\"' ����֤�����Ƿ�������"
