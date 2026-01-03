#!/usr/bin/env bash

# 视图 A：首字响应曲线 (决定用户爽感)
vllm bench sweep plot ./sweep_res \
    --var-x request_throughput \
    --var-y p99_ttft_ms \
    --curve-by model_id \
    --fig-name TTFT_Response_Time \
    --fig-dir ./img

# 视图 B：有效产出对比 (决定业务成本)
# 需要定义对应的request_goodput args: --goodput
vllm bench sweep plot ./sweep_res \
    --var-x request_throughput --var-y request_goodput --curve-by model_id \
    --fig-name Goodput_SLA_Analysis \
    --fig-dir ./img

# 视图 C：系统抖动分析 (决定系统稳定性)
vllm bench sweep plot ./sweep_res \
    --var-x request_throughput --var-y std_ttft_ms --curve-by model_id \
    --fig-name Latency_Stability \
    --fig-dir ./img
