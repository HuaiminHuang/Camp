import asyncio
"""
┌─────────────────────────┐
│      asyncio.run()      │  ← 事件循环生命周期
│   ┌─────────────────┐   │
│   │   async main()  │   │  ← 程序入口（唯一）
│   │   ┌───────────┐ │   │
│   │   │ await ... │ │   │  ← 协程调度
│   │   └───────────┘ │   │
│   └─────────────────┘   │
└─────────────────────────┘

"""

async def worker(i):
    await asyncio.sleep(1)
    return i

# gether
async def main_gather():
    tasks = [
        asyncio.create_task(worker(i))
        for i in range(5, 0, -1)
    ]

    results = await asyncio.gather(
        *tasks,
        return_exceptions=True
    )

    print("gater result", results)

# complete
async def main_complete():
    tasks = [
        asyncio.create_task(worker(i))
        for i in range(5, 0, -1)
    ]
    res = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        # print(result)
        res.append(result)
    print("complete result:", res)

if __name__ == "__main__":
    asyncio.run(main_gather())
    asyncio.run(main_complete())
