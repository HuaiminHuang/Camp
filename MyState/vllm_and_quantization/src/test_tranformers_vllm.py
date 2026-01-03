import torch
import time
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM

def test_transformers():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    local_path = "../Qwen/Qwen2.5-0.5B-Instruct"
    model = AutoModelForCausalLM.from_pretrained(local_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(local_path)

    prompt = "给我想一个关于人工智能改变世界的科幻故事大纲。"
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    start_time = time.time()
    output = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    end_time = time.time()

    print(f"Transformers 生成耗时: {end_time - start_time:.2f} 秒")
    print(tokenizer.decode(output[0], skip_special_tokens=True))

def test_vllm():
    local_path = "../Qwen/Qwen2.5-0.5B-Instruct"
    # 初始化 vLLM。注意：vLLM 默认会占用 90% 的显存，如果显存较小，可以设置 gpu_memory_utilization=0.5
    llm = LLM(model=local_path, gpu_memory_utilization=0.6)

    sampling_params = SamplingParams(temperature=0, max_tokens=512)
    prompt = "给我想一个关于人工智能改变世界的科幻故事大纲。"

    start_time = time.time()
    outputs = llm.generate([prompt], sampling_params) # 注意：vLLM 支持传入列表进行批处理
    end_time = time.time()

    for output in outputs:
        print(f"vLLM 生成耗时: {end_time - start_time:.2f} 秒")
        print(output.outputs[0].text)

if __name__ == "__main__":
    input("")
    # test_transformers()
    test_vllm()