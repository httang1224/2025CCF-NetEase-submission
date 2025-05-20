lm-eval --model vllm --model_args pretrained=./models/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto  --tasks arc_challenge --batch_size auto:1 --output_path ./outputs/
lm-eval --model vllm --model_args pretrained=./models/Llama-3.2-3B-Instruct/,gpu_memory_utilization=0.6,dtype=auto  --tasks gsm8k --batch_size auto:1 --output_path ./outputs/
