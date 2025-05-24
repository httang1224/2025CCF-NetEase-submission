# 面向大规模预训练模型的小型化部署优化方法研究

本项目致力于研究如何在保证精度的前提下，实现大模型在有限算力平台上的高效部署。我们以 Llama-3.2-3B-Instruct 为基准模型，采用 GPTQ和AWQ 进行量化优化，配合 `vLLM` 和 `lm-evaluation-harness` 进行精度与延迟评估。

## 1. 量化性能

<p align="center">
  <img src="./scripts/assets/gsm_arc.png" width="600"/>
</p>



---

## 2. 项目结构

```
.
├── benchmarks/                  # 推理延迟评估脚本
│   ├── benchmark_latency.py
│   └── benchmark_utils.py
├── models/
│   └── Llama-3.2-3B-Instruct/   # 原始/优化后模型路径
├── outputs/                     # 精度与延迟评估结果
│   ├── acc.json
│   └── perf.json
└── ...
```

---



## 3. 环境创建

```bash
git clone https://github.com/httang1224/2025CCF-NetEase-submission.git
git clone git@github.com:httang1224/2025CCF-NetEase-submission.git

cd 2025CCF-NetEase-submission

# 赋予可执行权限（如果还未执行）
chmod +x install_env.sh

# 运行脚本创建并配置环境
./install_env.sh
```

---



## 4. 原始模型准备

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --token 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 --resume-download \
  meta-llama/Llama-3.2-3B-Instruct \
  --local-dir meta-llama/Llama-3.2-3B-Instruct
```

---



## 5. 原始模型性能基准

### 5.1 推理延迟测试

```python
python3 ./benchmarks/benchmark_latency.py \
  --model ./meta-llama/Llama-3.2-3B-Instruct/ \
  --input-len 4096 --output-len 100 --batch-size 1
```

```
TTFT: 0.2929 s
TPOT: 0.0151 s
weights_memory: 6127.83 MB
```



### 5.2 精度评估

#### 5.2.1 GSM8K 

```bash
# 小学数学应用题，开放式问答，最终返回一个数值答案
lm-eval --model vllm --model_args pretrained=./meta-llama/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto \
  --tasks gsm8k --batch_size auto:1 --output_path ./outputs/
```

```
|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6543|±  |0.0131|
|     |       |strict-match    |     5|exact_match|↑  |0.6482|±  |0.0132|
```

#### 5.2.2 ARC-Challenge 

```bash
# 小学/初中科学考试题 	多选题（四选一）
lm-eval --model vllm --model_args pretrained=./meta-llama/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto \
  --tasks arc_challenge --batch_size auto:1 --output_path ./outputs/
```

```
|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.4352|±  |0.0145|
|             |       |none  |     0|acc_norm|↑  |0.4582|±  |0.0146|
```

---



## 6. 量化后模型评估

### 6.1 基于GPTQ

详见 `outputs/acc.json` 与 `outputs/perf.json` 文件，或运行以下脚本：

```bash
bash run_perf.sh   # 运行性能评估
bash run_acc.sh    # 运行精度评估
```

---

### 6.2  基于AWQ

详见 `outputs/acc.json` 与 `outputs/perf.json` 文件，或运行以下脚本：

```bash
bash run_perf.sh   # 运行性能评估
bash run_acc.sh    # 运行精度评估
```

---



## 7. 比赛评价指标说明

### 7.1 精度能力评估（Accuracy）

以 ARC_challenge 与 GSM8K 两个任务在 `lm-eval` 上的平均分作为衡量标准。

- 参考指标 = Llama-3.2-3B-Instruct 精度均值
- 提交模型指标 = 你优化后的模型精度均值

### 7.2 性能指标评估（Efficiency）

- **模型压缩率**：模型权重大小比值
- **推理速度提升**：
  - TTFT（Time to First Token）
  - TPOT（Time Per Output Token）

均与原始模型对比，计算提升率。

### 7.3 总分计算

所有精度与性能项均按参赛最高值归一化，进行加权求和得出最终得分。

---



## 8. 验证环境说明

> ⚠️ 注意：虽然项目评估需求标准基于 4090 GPU，但本实验与性能测试实则在 **NVIDIA A40 (48GB)** 上实测完成

- 平台：网易丹炉云平台
- GPU：RTX 4090
- CUDA：12.1
- vLLM：0.7.1
- PyTorch：2.5.1
- lm-eval：0.4.8

---

