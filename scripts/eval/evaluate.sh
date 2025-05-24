#!/bin/bash

# Default conda environment name
ENV_NAME=${1:-llm_compress}

echo ">>> Creating Conda environment: $ENV_NAME"

# Conda initialization
source "$(conda info --base)/etc/profile.d/conda.sh"

# Check if the environment already exists
if conda env list | grep -q "$ENV_NAME"; then
  echo ">>> Environment '$ENV_NAME' already exists. Skipping creation."
else
  conda create -n "$ENV_NAME" python=3.9 -y
fi

# Activate the environment
conda activate "$ENV_NAME"
echo ">>> Conda environment '$ENV_NAME' activated. Installing dependencies..."

# Install PyTorch (ensure vLLM compatibility)
pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121

# Install main dependencies
pip install auto-gptq==0.7.1 datasets==2.17.0 vllm==0.7.1 \
lm-eval==0.4.8 huggingface_hub autoawq nvitop ipykernel matplotlib seaborn

echo ""
echo ">>> All dependencies installed. Try running: python -c \"from vllm import LLM\" to verify."
