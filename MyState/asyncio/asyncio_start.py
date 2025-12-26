"""
异步编程示例：演示同步与异步执行的差异

异步编程通过"任务挂起与切换"，实现了在同一时间内处理多个 I/O 任务的能力，
避免了 CPU 在等待过程中的资源浪费。

本示例展示了：
1. 同步串行执行 vs 异步并发执行的性能对比
2. asyncio.gather() 的使用方法
3. asyncio.as_completed() 的使用方法
"""

import time
import asyncio
from typing import List, Union


def get_post(task_id: int) -> str:
    """同步获取数据函数
    
    模拟一个阻塞式的I/O操作，使用 time.sleep 进行阻塞等待。
    
    Args:
        task_id: 任务编号
        
    Returns:
        str: 任务结果字符串
    """
    print(f"任务 {task_id} 开始 (同步)...")
    
    # time.sleep 是阻塞式的，CPU 在这 2 秒内会完全停下等待
    time.sleep(2) 
    
    return f"结果 {task_id}"


def main_sync() -> None:
    """同步主函数：演示串行执行
    
    依次执行3个任务，总耗时为各任务耗时之和。
    """
    print("=== 开始同步串行测试 ===")
    total_start = time.perf_counter()

    # 依次执行 3 个任务
    results: List[str] = []
    for i in range(1, 4):
        res = get_post(i)
        results.append(res)

    total_end = time.perf_counter()
    print(f"=== 全部完成，总总耗时: {total_end - total_start:.2f} 秒 ===")
    print(f"返回结果: {results}\n")


async def get_post_asyncio(task_id: int) -> str:
    """异步获取数据函数
    
    模拟一个非阻塞的I/O操作，使用 asyncio.sleep 进行异步等待。
    
    Args:
        task_id: 任务编号
        
    Returns:
        str: 任务结果字符串
    """
    start_time = time.perf_counter()
    print(f"任务 {task_id} 开始...")
    
    await asyncio.sleep(2)  # 模拟网络IO，非阻塞式等待
    
    end_time = time.perf_counter()
    print(f"任务 {task_id} 完成，耗时: {end_time - start_time:.2f} 秒")
    return f"结果 {task_id}"


async def asyncio_main() -> None:
    """异步主函数：演示并发执行
    
    使用 asyncio.gather() 同时启动多个任务，总耗时约等于最长任务的耗时。
    """
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
    results = await asyncio.gather(
        *tasks, 
        return_exceptions=True  # 可选参数，默认：一个异常 --> 全部取消
    )

    """
    更显式的写法为：
    t1 = asyncio.create_task(get_post_asyncio(1))
    t2 = asyncio.create_task(get_post_asyncio(2))
    t3 = asyncio.create_task(get_post_asyncio(3))

    results = await asyncio.gather(t1, t2, t3)
    """
    total_end = time.perf_counter()

    print(f"=== 全部完成，总耗时: {total_end - total_start:.2f} 秒 ===")
    print(f"返回结果: {results}\n")


async def complete_first() -> None:
    """演示 asyncio.as_completed() 的使用
    
    asyncio.as_completed() 会按照任务完成的顺序返回结果，
    不保证按照任务创建的顺序。
    
    注意：原函数名 'complete_firt' 有拼写错误，已修正为 'complete_first'
    """
    # 创建5个任务，倒序创建
    tasks = [
        asyncio.create_task(get_post_asyncio(i))
        for i in range(5, 0, -1)
    ]

    start = time.perf_counter()
    results = []
    # 按完成顺序获取结果，不保证顺序
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)
    end = time.perf_counter()

    print(f"=== 全部完成，总耗时: {end - start:.2f} 秒 ===")
    print(f"返回结果: {results}\n")



if __name__ == "__main__":
    # 异步测试
    asyncio.run(asyncio_main())
    # 同步测试
    main_sync()

    """                                                       
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
    asyncio.run(complete_first())
    """
    任务 5 开始...
    任务 4 开始...
    任务 3 开始...
    任务 2 开始...
    任务 1 开始...
    任务 5 完成，耗时: 2.01 秒
    任务 3 完成，耗时: 2.01 秒
    任务 1 完成，耗时: 2.01 秒
    任务 4 完成，耗时: 2.01 秒
    任务 2 完成，耗时: 2.01 秒
    收到结果: 结果 5
    收到结果: 结果 3
    收到结果: 结果 1
    收到结果: 结果 4
    收到结果: 结果 2
    """
