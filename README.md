
# 🚀 面向大规模预训练模型的小型化部署优化方法

本项目旨在研究如何在保证精度的前提下，实现大模型在有限算力平台上的高效部署。以 **LLaMA-3.2-3B-Instruct** 为基准模型，分别采用了 **GPTQ** 和 **AWQ** 进行量化压缩，并结合 `vLLM` 与 `lm-evaluation-harness` 工具，对模型在 **准确率与推理延迟** 两方面进行系统评估。

---



## 📊 1. 量化性能概览

<p align="center">
  <img src="./assets/gsm_arc.png" width="750"/>
</p>
<table>
  <thead>
    <tr>
      <th>Model</th>
      <th>0515-1_INT8</th>
      <th>0519-3_INT4</th>
      <th>0526-1_INT4</th>
      <th>0605-1_INT4</th>
      <th>LLaMA</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><strong>AVG (%)</strong></td>
      <td>48.53</td>
      <td>54.22</td>
      <td>54.43</td>
      <td>54.45</td>
      <td>54.90</td>  
    </tr>
  </tbody>
</table>
<p align="center">
  <em>🧪  <strong>To be continued ...</strong></em>
</p>



```bash
# 更新时间 2025年6月5日20:22:16
# 最佳量化模型路径
./models/s961_int4_Llama-3.2-3B-Instruct/

# 模型权重百度网盘下载链接
https://pan.baidu.com/s/1XugFga2leGdZrHf1y8jZxQ?pwd=1224

# 量化模型评估范式
./scripts/eval/evaluate.sh ./models/s961_int4_Llama-3.2-3B-Instruct/
```




---

## 🗂️ 2. 项目结构

```bash
.
├── scripts/                         
│   ├── env/                         # 环境配置脚本
│   │   └── install_env.sh
│   │
│   ├── eval/                        # 评估脚本
│   │   └── evaluate.sh
│   │
│   ├── benchmarks/                 # 延迟评估代码
│   │   ├── benchmark_latency.py
│   │   └── benchmark_utils.py
│   │
│   └── quant/                      # 模型量化实现
│       ├── GPTQ_v1.py
│       └── AWQ_v1.py
│
├── models/                         # 原始与量化模型
│   ├── Llama-3.2-3B-Instruct/
│   └── Int4_gptq_v1/
│   └── s521_int4_Llama-3.2-3B-Instruct/    
│   └── s961_int4_Llama-3.2-3B-Instruct/    # best
│
├── outputs/                        	# 评估结果（精度/延迟）
│   └── s521_int4_Llama-3.2-3B-Instruct/
│   	├── latent.json
│   	├── gsm8k.json
│   	└── arc_challenge.json
│
└── README.md                       # 项目说明文档			 
```

---

## 🧱 3. 环境配置

```bash
git clone git@github.com:httang1224/2025CCF-NetEase-submission.git && cd 2025CCF-NetEase-submission

chmod +x ./scripts/env/install_env.sh (optional)

./scripts/env/install_env.sh llm_compress
conda activate llm_compress
```

---

## 📦 4. 模型下载

```bash
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --token hf_*** --resume-download \
  meta-llama/Llama-3.2-3B-Instruct \
  --local-dir ./models/Llama-3.2-3B-Instruct
```

---

## 🔬 5. 原始模型性能基准评估

### ⏱️ 推理延迟评估

```bash
python3 ./benchmarks/benchmark_latency.py \
  --model ./models/Llama-3.2-3B-Instruct/ \
  --input-len 4096 --output-len 100 --batch-size 1
```

```
TTFT: 0.292 seconds
TPOT: 0.015 seconds
weights_memory: 6127.833 MB
```

###  🧪精度评估

- GSM8K（小学数学应用题）：

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
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6543|±  |0.0131|
|     |       |strict-match    |     5|exact_match|↑  |0.6482|±  |0.0132|
```



- ARC-Challenge（科学选择题）：

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

###  🛠️ GPTQ 量化

```bash
python ./scripts/quant/GPTQ_v1.py
```

### 📏 AWQ 量化

```bash
python ./scripts/quant/AWQ_v1.py
```

### ⚙️ 量化评估

```bash
chmod +x ./scripts/eval/evaluate.sh (optional)

./scripts/eval/evaluate.sh ./models/Int8_gptq_v1
./scripts/eval/evaluate.sh ./models/Int4_gptq_v1
./scripts/eval/evaluate.sh ./models/Int4_awq_v1

./scripts/eval/evaluate.sh ./models/s521_int4_Llama-3.2-3B-Instruct/
./scripts/eval/evaluate.sh ./models/s961_int4_Llama-3.2-3B-Instruct/
```

---

## 🏆 7. 比赛评分指标说明

### 🎯 精度指标

模型在 `ARC-Challenge` 与 `GSM8K` 任务上的平均得分：

- **参考基线**：原始模型分数（LLaMA-3.2-3B-Instruct）
- **优化目标**：你优化后的压缩模型得分

### ⚡ 性能指标

-  模型压缩率（模型文件体积对比）
- 推理速度提升：
  - TTFT（Time to First Token）
  - TPOT（Time Per Output Token）

所有性能指标相较原始模型进行比值计算。

### 🧮 总评分计算

> 精度与性能指标将归一化加权，构成最终得分。

---

## 🖥️ 8. 实验硬件说明

> ⚠️ 当前实验评估结果基于 **NVIDIA A40 48GB**，官方推荐配置为 RTX 4090。

---

