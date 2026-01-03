#!/bin/bash
# 确保脚本有执行权限：chmod +x bench_sweep.sh

# ================================
DIR="sweep_res"
# 注意：确保这些路径是正确的
MODELS=("./Qwen3-0.6B-awq-sym" "./Qwen/Qwen3-0.6B")
PORT=8001
MAX_MODEL_LEN=4096
# ================================
mkdir -p "$DIR" logs

for model in "${MODELS[@]}"; do
    # 提取模型名称（如 Qwen3-0.6B）
    model_name=$(basename "$model")

    echo "========================================"
    echo "正在测试模型: $model_name"
    echo "========================================"

    # 1. 自动判断量化参数
    QUANT_ARG=""
    if [[ $model == *"awq"* || $model == *"sym"* ]]; then
        # 提示：如果是 vLLM 官方支持的压缩格式，请确认是 awq 还是 compressed-tensors
        QUANT_ARG="--quantization compressed-tensors"
    fi

    # 2. 定义启动命令
    SERVE_COMMAND="vllm serve $model \
        $QUANT_ARG \
        --host 127.0.0.1 \
        --port $PORT \
        --gpu-memory-utilization 0.7 \
        --max-model-len $MAX_MODEL_LEN \
        --dtype auto \
        --enforce-eager"

    # 3. 定义测试命令 (指定数据集和单次测试请求数)
    # sweep 会自动根据 --request-rate 覆盖测试速率
    BENCH_COMMAND="vllm bench serve \
        --port $PORT \
        --model $model \
        --dataset-name random \
        --num-prompts 200 \
        --save-result"

    # 4. 执行全自动扫描
    vllm bench sweep serve \
        --serve-cmd "$SERVE_COMMAND" \
        --bench-cmd "$BENCH_COMMAND" \
        --bench-params ./params.json \
        --output-dir "./$DIR/sweep_${model_name}_results" \
        2>&1 | tee -a "logs/benchmark_through_${model_name}.log"
done