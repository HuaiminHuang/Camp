from openai import OpenAI
import argparse

def main(prompt):
    # 这里的端口和启动时的 8001 对应
    client = OpenAI(
        base_url="http://localhost:8001/v1",
        api_key="token-vllm" 
    )

    completion = client.chat.completions.create(
    model="Qwen/Qwen2.5-0.5B-Instruct", # 必须和 vllm 启动时的模型名一致
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": prompt}
    ],
    temperature=0.7
    )

    print(completion.choices[0].message.content)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--prompt", required=True, help="用户问题")
    args = parser.parse_args()
    main(args.prompt)