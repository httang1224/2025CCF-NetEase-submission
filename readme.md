# 面向大规模预训练模型的小型化部署优化方法研究


```
2025CCF---NetEase-Leihuo-Joint-Fund-Project-Database/
├── code/
│   ├── run_acc.py            # 精度评测脚本（调用 lm-eval）
│   └── run_perf.sh           # 性能评测脚本（TTFT, TPOT）
│
├── model/
│   ├── llama3-int/      # 模型目录（或存链接）
│   └── README.md           
│
├── demo/
│   ├── acc.json              # 精度评估结果（gsm8k / arc）
│   ├── perf.json             # 性能评估结果（推理速度等）
│   └── run_log.txt           # 可选：推理运行 log 记录
│
├── readme/
│   ├── result_plot.pdf       # 最终对比图
│   └── submission_report.md  # 报告说明：方法 + 实验结果
│
└── README.md                 # 仓库主页说明（简要描述项目）

```

### 1. 环境创建

```bash
# 创建新的 Conda 环境（可根据需要修改名称）
conda create -n llm_compress python=3.9 -y
conda activate llm_compress

pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
pip install auto-gptq==0.7.1
pip install datasets==2.17.0
pip install vllm==0.7.1
pip install lm-eval==0.4.8
```



### 2 下载原始模型

```bash
# Llama-3.2-3B-Instruct

export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download --token hf_******** --resume-download meta-llama/Llama-3.2-3B-Instruct --local-dir meta-llama/Llama-3.2-3B-Instruct
```



### 3 原始模型指标

```bash
# latent
python3 ./benchmarks/benchmark_latency.py --model /home/u2024030061/ht/hf_weight/meta-llama/Llama-3.2-3B-Instruct/  --input-len 4096 --output-len 100 --batch-size 1


TTFT: 0.2928596447221935 seconds
TPOT: 0.015151367858303113 seconds
weights_memory: 6127.833984375 MB
```





```bash
# gsm 
lm-eval --model vllm --model_args pretrained=/home/u2024030061/ht/hf_weight/meta-llama/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto  --tasks gsm8k --batch_size auto:1 --output_path ./outputs/

|Tasks|Version|     Filter     |n-shot|  Metric   |   |Value |   |Stderr|
|-----|------:|----------------|-----:|-----------|---|-----:|---|-----:|
|gsm8k|      3|flexible-extract|     5|exact_match|↑  |0.6543|±  |0.0131|
|     |       |strict-match    |     5|exact_match|↑  |0.6482|±  |0.0132|
```




```bash
# arc_challenge 
lm-eval --model vllm --model_args pretrained=/home/u2024030061/ht/hf_weight/meta-llama/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto  --tasks arc_challenge --batch_size auto:1 --output_path ./outputs/

|    Tasks    |Version|Filter|n-shot| Metric |   |Value |   |Stderr|
|-------------|------:|------|-----:|--------|---|-----:|---|-----:|
|arc_challenge|      1|none  |     0|acc     |↑  |0.4352|±  |0.0145|
|             |       |none  |     0|acc_norm|↑  |0.4582|±  |0.0146|
```



### 4 量化后指标

