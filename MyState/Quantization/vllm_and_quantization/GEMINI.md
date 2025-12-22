# Gemini Code Assistant Context: LLM Quantization and Benchmarking

## Project Overview

This project is designed to quantize and benchmark the performance of Large Language Models (LLMs). It focuses on comparing the original `Qwen/Qwen3-0.6B` model with its 4-bit AWQ (Activation-aware Weight Quantization) quantized version.

The primary tools used are:
- **vLLM**: For high-performance serving and throughput/latency benchmarking.
- **llmcompressor**: For performing one-shot AWQ quantization.
- **Hugging Face `transformers`**: For loading models and tokenizers.

The directory contains scripts for quantization, serving, benchmarking, and plotting the results.

## Key Files & Directories

- `src/`: Directory containing all Python scripts for quantization and model interaction.
  - `llm_awq_copressed.py`: Performs one-shot AWQ quantization.
- `scripts/`: Directory containing all shell scripts for automation.
  - `benchmark_through.sh`: Runs `vllm bench throughput` to compare model performance.
  - `vllm_servers.sh`: Launches a vLLM server for a specific model.
- `params.json`: A configuration file containing different request rates for benchmarking.
- `Qwen/Qwen3-0.6B/`: Directory containing the original, pre-trained FP16 model.
- `Qwen3-0.6B-awq-sym/`: Directory containing the 4-bit AWQ quantized model.
- `logs/`: Directory where log files from benchmarks and servers are stored.
- `sweep_res/`: Directory where benchmark results and generated plots are stored.

## Core Workflows

### 1. Model Quantization

The `src/llm_awq_copressed.py` script handles the quantization process. It uses the `llmcompressor` library to apply the AWQ algorithm.

**To run quantization (example):**
```bash
# This is an illustrative command based on the script's content
python src/llm_awq_copressed.py
```
This will create a new directory (e.g., `Qwen3-0.6B-awq-sym`) containing the compressed model files.

### 2. Benchmarking

The `scripts/benchmark_through.sh` script is the main entry point for performance benchmarking. It uses `vllm bench throughput`.

**To run the throughput benchmark:**
```bash
bash scripts/benchmark_through.sh
```
The script will iterate through the models defined in its `MODELS` array and log the output to `logs/benchmark_through.log`.

### 3. Serving Models with vLLM

The `scripts/vllm_servers.sh` script provides a template for running a vLLM server. The commands should be run from the project's root directory.

**To serve the original model:**
```bash
# Modify scripts/vllm_servers.sh to point to the correct model path and remove the QUANT_ARG
vllm serve ./Qwen/Qwen3-0.6B --host 127.0.0.1 --port 8001
```

**To serve the AWQ quantized model:**
It is critical to include the `--quantization compressed-tensors` flag.
```bash
# Modify scripts/vllm_servers.sh to point to the quantized model and ensure QUANT_ARG is set
vllm serve ./Qwen3-0.6B-awq-sym \
    --host 127.0.0.1 \
    --port 8001 \
    --quantization compressed-tensors
```

## Development Conventions

- Shell scripts in the `scripts/` directory are used to automate workflows.
- Python scripts in the `src/` directory are used for the core logic.
- All scripts should be run from the project root directory.
- Model paths are hardcoded in the scripts. To test different models, you will need to modify the script variables.
- Quantized models are expected to have `awq` or `sym` in their name for the scripts to automatically apply the correct quantization arguments.
