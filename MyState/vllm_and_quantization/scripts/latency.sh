#!/usr/bin/env bash
# ------------------------------------------
# 首次启动
# chmod +x latency.sh
# 基础参数设置
MODEL="Qwen/Qwen3-0.6B"   # 模型名或本地路径
# MODEL="./Qwen3-0.6B-awq-sym"   # 模型名或本地路径
INPUT_LEN=(128 1024 2048)   # 输入长度（token）
OUTPUT_LEN=128              # 输出长度（token）
BATCH_SIZE=(1 8 16 32)      # 一次喂给引擎的序列数
NUM_ITERS=30                # 正式迭代轮次
WARMUP=10                   # warmup 轮次
MAX_MODEL_LEN=4096          # 模型上下文窗口
# QUANTIZATION=compressed-tensors          # 量化类型
# ------------------------------------------
# 以下一般不动
DTYPE="auto"              # float16 / bfloat16 / auto
TP=1                      # 张量并行卡数
SEED=42                   # 随机种子
# ------------------------------------------
echo "==== bench latency ===="
echo "model      : $MODEL"
echo "batch      : $BATCH_SIZE"
echo "in/out len : $INPUT_LEN / $OUTPUT_LEN"
echo "iters      : $WARMUP(warmup)+$NUM_ITERS"
echo "======================"

# 强制使用 V0 引擎避免 Pydantic 校验报错（如果还没改 config.json）
# export VLLM_USE_V1=0
# 原生awq版本的权重
# "weights": {
#           "actorder": null,
#           "block_structure": null,
#           "dynamic": false,
#           "group_size": 128,
#           "num_bits": 4,
#           "observer": "minmax",
#           "observer_kwargs": {},
#           "scale_dtype": null,
#           "strategy": "group",
#           "symmetric": true,
#           "type": "int",
#           "zp_dtype": null
#         }
# 当前 VllmConfig 无效参数["zp_dtype", "scale_dtype"] 需要删除
# 版本差异：
# (quantization_llm) h2mzzz@Mk:~/quantization$  pip show compressed-tensors | grep Version --> Version: 0.13.0
# (vllm) h2mzzz@Mk:~/vllm_and_quantization$ pip show compressed-tensors | grep Version --> Version: 0.12.2
echo "开始多维度性能测试..."
echo "Model: $MODEL"
echo "------------------------------------------------------"

for in_len in "${INPUT_LEN[@]}"; do
    for bs in "${BATCH_SIZE[@]}"; do
        echo "[测试中] Input_Len: $in_len | Batch_Size: $bs"

        vllm bench latency \
        --model "$MODEL" \
        --dtype "$DTYPE" \
        --tensor-parallel-size "$TP" \
        --batch-size "$bs" \
        --input-len "$in_len" \
        --output-len "$OUTPUT_LEN" \
        --num-iters-warmup "$WARMUP" \
        --num-iters "$NUM_ITERS" \
        --seed "$SEED" \
        --gpu-memory-utilization 0.8 \
        --max-model-len "$MAX_MODEL_LEN" \
        --quantization "$QUANTIZATION" >> ./logs/benchmark_results_origin_latency.log 2>&1

        # 简单检查上一个命令是否崩溃（例如显存溢出）
        if [ $? -ne 0 ]; then
            echo "!!! 警告: BS=$bs, In=$in_len 组合可能导致 OOM 或错误，跳过。"
            continue
        fi
    done
done
