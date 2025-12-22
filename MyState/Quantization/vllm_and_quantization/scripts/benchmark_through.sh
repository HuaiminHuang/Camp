#!/usr/bin/env bash

# 1. 定义数组
MODELS=("../Qwen3-0.6B-awq-sym" "../Qwen/Qwen3-0.6B")
INPUT_LENS=(128 1024 2048)  # 修正变量名
PROMPTS=200
OUTPUT_LEN=128
MAX_MODEL_LEN=4096

# mkdir -p logs  # 确保目录存在

for M in "${MODELS[@]}"; do
    echo "========================================"
    echo "正在测试模型: $M"
    echo "========================================"

    for len in "${INPUT_LENS[@]}"; do
        echo "[测试中] 输入长度: $len | 请求总数: $PROMPTS"
        
        # 确定量化参数（如果是原版模型，不传量化参数）
        QUANT_ARG=""
        if [[ $M == *"awq"* || $M == *"sym"* ]]; then
            QUANT_ARG="--quantization compressed-tensors"
        fi

        # 顺序执行，不加 &
        vllm bench throughput \
            --model "$M" \
            --max_model_len "$MAX_MODEL_LEN" \
            --dataset-name "random" \
            --input-len "$len" \
            --output-len "$OUTPUT_LEN" \
            --num-prompts "$PROMPTS" \
            $QUANT_ARG \
            --trust-remote-code \
            --gpu-memory-utilization 0.8 \
            --dtype auto 2>&1 | tee -a ../logs/benchmark_through.log

        echo -e "\n" # 换行
    done
done