
# 🚀 面向大规模预训练模型的小型化部署优化方法

本项目旨在研究如何在保证精度的前提下，实现大模型在有限算力平台上的高效部署。我们以 **LLaMA-3.2-3B-Instruct** 为基准模型，采用 **GPTQ** 和 **AWQ** 进行量化压缩，并结合 `vLLM` 与 `lm-evaluation-harness` 工具，对模型在 **准确率与推理延迟** 两方面进行系统评估。

---



## 📊 1. 量化性能优化

<p align="center">
  <img src="./assets/gsm_arc.png" width="1000"/>
</p>

<p align="center">
  <em>🧪  <strong>To be continued ...</strong></em>
</p>


---

## 🗂️ 2. 项目结构

```
.
├── scripts/benchmarks/            # 推理延迟评估
│   ├── benchmark_latency.py
│   └── benchmark_utils.py
│
├── models/                        # 原始/量化模型存放目录
│   └── Llama-3.2-3B-Instruct/
│   └── Int4_gptq_v1/
│
├── outputs/                       # 精度与性能评估结果
│   ├── acc.json
│   └── perf.json
│
├── install_env.sh					# 安装环境   				        
└── evaluate.sh						# 量化评估				 
```

---

## 🧱 3. 环境配置与依赖安装

```bash
git clone git@github.com:httang1224/2025CCF-NetEase-submission.git
cd 2025CCF-NetEase-submission

chmod +x install_env.sh
./install_env.sh
conda activate llm_compress
```

---

## 📦 4. 原始模型下载与准备

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --token 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 --resume-download \
  meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/Llama-3.2-3B-Instruct
```

---

## 🔬 5. 原始模型性能基准评估

### ⏱️ 5.1 推理延迟评估

```bash
python3 ./benchmarks/benchmark_latency.py \
  --model ./models/Llama-3.2-3B-Instruct/ \
  --input-len 4096 --output-len 100 --batch-size 1
```

```
TTFT: 0.2929 s
TPOT: 0.0151 s
weights_memory: 6127.83 MB
```

### 🧪 5.2 精度评估

#### 📘 GSM8K：小学数学应用题

```bash
lm-eval --model vllm \
  --model_args pretrained=./models/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto \
  --tasks gsm8k \
  --batch_size auto:1 \
  --output_path ./outputs/
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.0205|±  |0.0039|
|     |       |strict-match    |     5|exact_match|↑  |0.0000|±  |0.0000|
```

#### 🧪 ARC-Challenge：科学选择题

```bash
lm-eval --model vllm \
  --model_args pretrained=./models/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto \
  --tasks arc_challenge \
  --batch_size auto:1 \
  --output_path ./outputs/
```

```
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.4352|±  |0.0145|
|             |       |none  |     0|acc_norm|↑  |0.4582|±  |0.0146|
```

---

## 🔧 6. 量化模型评估

### ⚙️ 6.1 使用 GPTQ 进行量化

```bash
python GPTQ_v1.py
```

### 🧪 6.2 量化模型精度评估

```bash
chmod +x evaluate.sh
./evaluate.sh ./models/Int8_gptq_v1
```

### 🛠️ 6.3 使用 AWQ 进行量化（waiting）

```bash
# TODO: 添加 AWQ 量化流程
```

---

## 🏆 7. 比赛评分指标说明

### 🎯 7.1 精度指标（Accuracy）

模型在 `ARC-Challenge` 与 `GSM8K` 任务上的平均得分：

- 📊 **参考基线**：原始模型分数（LLaMA-3.2-3B-Instruct）
- 🚀 **优化目标**：你优化后的压缩模型得分

### ⚡ 7.2 性能指标（Efficiency）

- 📉 模型压缩率（模型文件体积对比）
- ⚡ 推理速度提升：
  - TTFT（Time to First Token）
  - TPOT（Time Per Output Token）

所有性能指标相较原始模型进行比值计算。

### 🧮 7.3 总分计算方式

> 所有指标将归一化后加权求和，作为最终得分。

---

## 🖥️ 8. 实验硬件说明

> ⚠️ 虽然官方推荐使用 RTX 4090，本项目评估实测平台为 **NVIDIA A40 48GB**，确保评估数据稳定、可复现。

---
