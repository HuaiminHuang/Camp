#!/bin/bash
# pkill -f vllm
# 确保：chmod +x vllm_servers.sh
QUANT_ARG=""
if [[ $M == *"awq"* || $M == *"sym"* ]]; then
    QUANT_ARG="--quantization compressed-tensors"
fi
MODEL=(../Qwen3-0.6B-awq-sym ../Qwen/Qwen3-0.6B)
vllm serve ../Qwen3-0.6B-awq-sym \
    --host 127.0.0.1 \
    --port 8001 \
    $QUANT_ARG \
    --gpu-memory-utilization 0.7 \
    --max-model-len 2048 \
    --dtype float16 \
    --enforce-eager > ../logs/vllm.log 2>&1 &