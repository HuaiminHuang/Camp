# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer

# local_path = "../Qwen/Qwen3-0.6B"
local_path = "../Qwen3-0.6B-awq-sym"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)


def main():
    # Create an LLM.
    llm = LLM(model=local_path, max_model_len=4096, gpu_memory_utilization=0.7)
    history = []
    while 1:
        query = input("\n输入：")
        if query.lower() in ["q", "退出", "exit", "quit"]:
            break
        if not query.strip():
            continue
        history.append({"role": "user", "content": query})
        # chat tmpelate
        prompt = f"<|im_start|>user\n{query}<|im_end|>\n<|im_start|>assistant\n"

        outputs = llm.generate(prompt, sampling_params)
        # Print the outputs.
        print("\nGenerated Outputs:\n" + "-" * 60)
        for output in outputs:
            prompt = output.prompt
            generated_text = output.outputs[0].text
            print(f"Output:    {generated_text!r}")
            print("-" * 60)
        history.append({"role": "assistant", "content": generated_text})


def chat():
    # 初始化模型
    llm = LLM(model=local_path, max_model_len=4096, gpu_memory_utilization=0.7)
    
    # 初始化历史记录，可以在这里加一个 System Prompt
    history = [{"role": "system", "content": "你是一个乐于助人的 AI 助手。"}]
    
    while True:
        query = input("\n输入：")
        if query.lower() in ["q", "退出", "exit", "quit"]:
            break
        if not query.strip():
            continue
            
        # 1. 将用户输入加入历史
        history.append({"role": "user", "content": query})

        # 2. 使用 llm.chat 而不是 generate
        # vLLM 会自动根据模型目录下的 tokenizer_config.json 渲染模板
        outputs = llm.chat(history, sampling_params)

        # 3. 提取结果
        generated_text = outputs[0].outputs[0].text
        
        print("\nAI 回答:" + "-" * 40)
        print(generated_text)
        print("-" * 47)

        # 4. 关键：将 AI 的回复也加入历史，实现记忆
        history.append({"role": "assistant", "content": generated_text})


if __name__ == "__main__":
    # main()
    chat()