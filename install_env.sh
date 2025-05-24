#!/bin/bash

echo "?? 正在创建 Conda 环境：llm_compress"

# 创建 Conda 环境
conda create -n llmcompress python=3.9 -y

# 初始化 Conda 环境（确保脚本中能使用 conda activate）
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate llmcompress

echo "? Conda 环境已激活，开始安装依赖..."

# 安装 PyTorch（CUDA 12.1）版本
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# 安装其他 Python 依赖
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
echo "所有依赖安装完成！"
