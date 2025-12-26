# https://github.com/openai/openai-python
# 详情参考官方文档
import asyncio
import time
from openai import AsyncOpenAI
import os
from dotenv import load_dotenv

# 配置 api key
load_dotenv("../../apikey.env")
key = os.getenv("DEEPSEEK_API_KEY")
async_client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")
MODEL = "deepseek-chat" 

# asyncio 还可以支持流式输出 
async def stream_async(prompt):
    response = await async_client.chat.completions.create(
        model=MODEL,
        messages=prompt,
        stream=True,
    )

    # 流式输出
    full_content = ""
    async for chunk in response:
        # if chunk.choices is None:
        #     continue
        content = chunk.choices[0].delta.content
        if content:
            # 逐字打印 flush
            print(content, end="", flush=True)
            full_content += content
    return full_content

async def summary_conversation(prompt):
    response = await async_client.chat.completions.create(
        model=MODEL,
        messages=prompt,
    )
    return response.choices[0].message.content

def chat():
    history = [{"role": "system", "content": "你是一个乐于助人的助手。"}]
    while True:
        print("="*50)
        print("请输入聊天消息(退出：q/exit)")
        print("="*50)
        prompt = input("聊天框：")
        if prompt.lower() in ["q", "exit"]:
            print("退出程序......")
            break

        if prompt is None:
            continue
        history.append(
            {"role": "user", "content": prompt}
        )

        if len(history) >= 20:
            print("\n系统提示：[正在压缩长记忆...]")
            history.append(
            {"role": "user", "content": "请简要总结以上对话的核心要点，以便后续参考。"}
        )
            summary = asyncio.run(summary_conversation(history))
            history = [
                {"role": "system", "content": "你是一个乐于助人的助手。"},
                {"role": "user", "content": f"这是我们之前的对话总结: {summary}"},
            ]
            print("系统提示：[记忆压缩完成]")

        print("AI: ", end="")
        completed_content = asyncio.run(stream_async(history))
        history.append({"role": "assistant", "content": completed_content})
        print()

if __name__ == "__main__":
    chat()