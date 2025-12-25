import time
import asyncio

# 同步函数
def get_post(task_id):
    print(f"任务 {task_id} 开始 (同步)...")
    
    # time.sleep 是阻塞式的，CPU 在这 2 秒内会完全停下等待
    time.sleep(2) 
    
    return f"结果 {task_id}"

def main_sync():
    print("=== 开始同步串行测试 ===")
    total_start = time.perf_counter()

    # 依次执行 3 个任务
    results = []
    for i in range(1, 4):
        res = get_post(i)
        results.append(res)

    total_end = time.perf_counter()
    print(f"=== 全部完成，总总耗时: {total_end - total_start:.2f} 秒 ===")
    print(f"返回结果: {results}")

# 显式异步函数
async def get_post_asyncio(task_id):
    start_time = time.perf_counter()
    print(f"任务 {task_id} 开始...")
    
    await asyncio.sleep(2)  # 模拟网络IO
    
    end_time = time.perf_counter()
    print(f"任务 {task_id} 完成，耗时: {end_time - start_time:.2f} 秒")
    return f"结果 {task_id}"

async def asyncio_main():
    print("=== 开始异步并发测试 ===")
    total_start = time.perf_counter()

    # 同时启动 3 个任务
    # 而不是一个个 await get_post_asyncio()
    tasks = [
        get_post_asyncio(1),
        get_post_asyncio(2),
        get_post_asyncio(3)
    ]
    
    # gather 会并发运行任务并等待它们全部完成
    results = await asyncio.gather(*tasks)

    total_end = time.perf_counter()

    print(f"=== 全部完成，总耗时: {total_end - total_start:.2f} 秒 ===")
    print(f"返回结果: {results}")

if __name__ == "__main__":
    # 异步测试
    asyncio.run(asyncio_main())
    # 同步测试
    main_sync()

    """
    cmd                                                                
    === 开始异步并发测试 ===
    任务 1 开始...
    任务 2 开始...
    任务 3 开始...
    任务 1 完成，耗时: 2.01 秒
    任务 2 完成，耗时: 2.01 秒
    任务 3 完成，耗时: 2.01 秒
    === 全部完成，总耗时: 2.01 秒 ===
    返回结果: ['结果 1', '结果 2', '结果 3']
    === 开始同步串行测试 ===
    任务 1 开始 (同步)...
    任务 2 开始 (同步)...
    任务 3 开始 (同步)...
    === 全部完成，总总耗时: 6.02 秒 ===
    返回结果: ['结果 1', '结果 2', '结果 3']

    """