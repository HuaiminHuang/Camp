#!/usr/bin/env bash

# 1. 基础刻画：分析用户体感（TTFT）
# 这是最常用的命令，它能帮你找到服务器在保证响应速度的前提下，最高能扛住多少 QPS。
vllm bench sweep plot ./sweep_res \
    --var-x request_throughput \
    --var-y p99_ttft_ms \
    --curve-by model_id \
    --bin-by "request_throughput%1" \
    --filter-by "p99_ttft_ms<1000" \
    --fig-dpi 300 \
    --fig-name Qwen3_0.6B_Final_Analysis

# 2. 进阶刻画：分析吐字流畅度（TPOT）
vllm bench sweep plot ./sweep_res \
    --var-x request_throughput \
    --bin-by "request_throughput%1" \
    --var-y mean_tpot_ms \
    --curve-by model_id \
    --fig-name tpot_analysis

