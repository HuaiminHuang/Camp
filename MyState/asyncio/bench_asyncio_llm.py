import asyncio
import time
import os
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv("../../apikey.env")

client = AsyncOpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

MODEL = "deepseek-chat"

# 单轮 streaming 调用（统计 TTFT & 总时长）
async def stream_chat(messages, session_id, round_id):
    start = time.perf_counter()
    first_token_time = None
    content = ""

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        stream=True,
    )

    async for chunk in response:
        delta = chunk.choices[0].delta.content
        if delta:
            if first_token_time is None:
                first_token_time = time.perf_counter()
            content += delta

    end = time.perf_counter()

    return {
        "session": session_id,
        "round": round_id,
        "ttft": first_token_time - start if first_token_time else None,
        "latency": end - start,
        "content": content,
    }

# 一个连续对话 Session（顺序执行）
async def conversation_session(session_id, rounds=3):
    history = [
        {"role": "system", "content": "你是一个乐于助人的助手。"}
    ]

    stats = []

    for i in range(rounds):
        user_msg = f"这是第 {i+1} 轮对话，请简要回答。"
        history.append({"role": "user", "content": user_msg})

        result = await stream_chat(history, session_id, i + 1)
        history.append({"role": "assistant", "content": result["content"]})

        stats.append(result)

    return stats

async def benchmark(concurrency=5, rounds=3):
    print(f"\n=== 并发 Session 数: {concurrency}, 每个 {rounds} 轮 ===\n")

    start = time.perf_counter()

    tasks = [
        asyncio.create_task(conversation_session(i, rounds))
        for i in range(concurrency)
    ]

    results = await asyncio.gather(*tasks)
    print(results)

    end = time.perf_counter()

    flat = [item for session in results for item in session]

    avg_latency = sum(x["latency"] for x in flat) / len(flat)
    avg_ttft = sum(x["ttft"] for x in flat if x["ttft"]) / len(flat)

    print("====== 性能统计 ======")
    print(f"总请求数: {len(flat)}")
    print(f"总耗时: {end - start:.2f}s")
    print(f"平均 RT: {avg_latency:.2f}s")
    print(f"平均 TTFT: {avg_ttft:.2f}s")
    print(f"吞吐: {len(flat)/(end-start):.2f} req/s")

if __name__ == "__main__":
    asyncio.run(
        benchmark(concurrency=5, rounds=3)
    )

"""
=== 并发 Session 数: 5, 每个 3 轮 ===

====== 性能统计 ======
总请求数: 15
总耗时: 5.94s
平均 RT: 1.77s
平均 TTFT: 1.12s
吞吐: 2.53 req/s
"""