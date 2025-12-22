#!/bin/bash

# 生成帕累托前沿可视化图表的Shell脚本
# 用于分析vLLM基准测试结果中的性能与延迟权衡

# 设置结果目录
RESULTS_DIR="sweep_res/sweep_Qwen3-0.6B-awq-sym_results"
OUTPUT_DIR="${RESULTS_DIR}/pareto"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

echo "正在生成帕累托前沿可视化图表..."
echo "结果目录: $RESULTS_DIR"
echo "输出目录: $OUTPUT_DIR"
echo ""

# 1. 生成基本的帕累托前沿图
# echo "1. 生成基本帕累托前沿图..."
# vllm bench sweep plot_pareto "$RESULTS_DIR" \
#     --label-by max_concurrency,gpu_count

# # 2. 生成带有不同标注的帕累托前沿图
# echo "2. 生成带有请求率标注的帕累托前沿图..."
# vllm bench sweep plot_pareto "$RESULTS_DIR" \
#     --label-by request_rate,max_concurrency

# 3. 生成带有吞吐量标注的帕累托前沿图
echo "3. 生成带有吞吐量标注的帕累托前沿图..."
vllm bench sweep plot_pareto "$RESULTS_DIR" \
    --label-by mean_ttft_ms

# # 4. 生成带有延迟标注的帕累托前沿图
# echo "4. 生成带有延迟标注的帕累托前沿图..."
# vllm bench sweep plot_pareto "$RESULTS_DIR" \
#     --label-by mean_ttft_ms,p99_e2el_ms

echo ""
echo "帕累托前沿图表生成完成！"
echo "图表保存在: $OUTPUT_DIR"
echo ""
echo "生成的图表文件："
ls -la "$OUTPUT_DIR"/*.png

echo ""
echo "帕累托前沿图显示了每用户令牌/秒与每GPU令牌/秒之间的权衡关系"
echo "前沿上的点代表了最优的性能配置，可以帮助您找到最佳的性能平衡点"