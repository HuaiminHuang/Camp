from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

"""
版本依赖问题
# from transformers.activations import NewGELUActivation, PytorchGELUTanh, GELUActivation
from transformers.activations import NewGELUActivation, GELUTanh, GELUActivation
# 然后起个别名，兼容 awq 后面的代码调用
PytorchGELUTanh = GELUTanh
"""

model_path = "../Qwen/Qwen3-0.6B"
quant_path = "../Qwen/Qwen3-0.6B_awq"
quant_config = {"zero_point": True, "q_group_size": 128, "w_bit": 4, "version": "GEMM"}

# Load model
model = AutoAWQForCausalLM.from_pretrained(
    model_path,
    safetensors=True,
    low_cpu_mem_usage=True,
    use_cache=False,
)
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(model)

# # Quantize
# model.quantize(tokenizer, quant_config=quant_config)

# # Save quantized model
# model.save_quantized(quant_path)
# tokenizer.save_pretrained(quant_path)

# print(f'Model is quantized and saved at "{quant_path}"')


# """
# bash cmd
# python examples/offline_inference/llm_engine_example.py \
#     --model TheBloke/Llama-2-7b-Chat-AWQ \
#     --quantization awq
# """