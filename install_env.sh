#!/bin/bash

echo "?? 正在创建 Conda 环境：llm_compress"

# Conda 初始化
source "$(conda info --base)/etc/profile.d/conda.sh"

# 判断环境是否已存在
if conda env list | grep -q "llm_compress"; then
  echo "?? 环境 llm_compress 已存在，跳过创建..."
else
  conda create -n llm_compress python=3.9 -y
fi

conda activate llm_compress
echo "? Conda 环境已激活，开始安装依赖..."

# 安装 PyTorch（注意兼容 vllm）
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 安装主要依赖
pip install auto-gptq==0.7.1 datasets==2.17.0 vllm==0.7.1 \
lm-eval==0.4.8 huggingface_hub autoawq nvitop ipykernel matplotlib seaborn

echo ""
echo "?? 所有依赖安装完成！建议运行 'python -c \"from vllm import LLM\"' 以验证环境是否正常。"
