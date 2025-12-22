#!/bin/bash

# 确保脚本有执行权限：chmod +x benchmark_basic.sh

# 测试服务压力
MODELS=("./Qwen3-0.6B-awq-sym" "./Qwen/Qwen3-0.6B")

for M in "{$MODEL[@]}"; do
    for rps in 1 4 8 10 16; do
        vllm bench serve \
        --backend vllm \
        --host 127.0.0.1 \
        --port 8001 \
        --model "$M" \
        --dataset-name random \
        --num-prompts 100 \
        --request-rate $rps \
        --random-input-len 512 \
        --random-output-len 128
    done
done