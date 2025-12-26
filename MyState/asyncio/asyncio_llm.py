import asyncio
import time
from openai import OpenAI, AsyncOpenAI
import os
from dotenv import load_dotenv

# 配置 api key
load_dotenv("../../apikey.env")
key = os.getenv("DEEPSEEK_API_KEY")
client = OpenAI(api_key=key, base_url="https://api.deepseek.com")
async_client = AsyncOpenAI(api_key=key, base_url="https://api.deepseek.com")

MODEL = "deepseek-chat" # 或 gpt-4
QUESTIONS = ["请写一首关于秋天的诗"] * 3  # 测试 3 次相同请求

# 同步测试
def test_sync():
    print("\n--- 开始同步调用测试 ---")
    start = time.perf_counter()
    results = []
    
    for i, q in enumerate(QUESTIONS):
        print(f"同步请求 {i+1} 发送中...")
        response = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": q}
            ]
        )
        results.append(response.choices[0].message.content)
        print(f"同步请求 {i+1} 已完成")

    end = time.perf_counter()
    print(f"同步总耗时: {end - start:.2f} 秒")
    print(*results, sep="\n")

# 异步测试
async def fetch_async(i, q):
    results = []
    print(f"异步请求 {i+1} 已发出...")
    response = await async_client.chat.completions.create(
        model=MODEL,
        messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": q}
            ]
    )
    print(f"异步请求 {i+1} 响应返回")
    

    return response.choices[0].message.content

async def test_async():
    print("\n--- 开始异步调用测试 ---")
    start = time.perf_counter()
    
    # 同时创建多个任务
    tasks = [fetch_async(i, q) for i, q in enumerate(QUESTIONS)]
    
    # 并发执行
    results = await asyncio.gather(*tasks)
    
    end = time.perf_counter()
    print(f"异步总耗时: {end - start:.2f} 秒")
    print(*results, sep="\n")

if __name__ == "__main__":
    # 运行同步测试
    test_sync()
    
    # 运行异步测试
    asyncio.run(test_async())

    """
    python asyncio_llm.py

    --- 开始同步调用测试 ---
    同步请求 1 发送中...
    同步请求 1 已完成
    同步请求 2 发送中...
    同步请求 2 已完成
    同步请求 3 发送中...
    同步请求 3 已完成
    同步总耗时: 16.80 秒
    《秋窗》
    西风卷地启云扃，忽有轻寒透画棂。
    千山删繁留瘦骨，一江澄澈褪虚形。
    雁字裁天书次第，虫声织夜补空灵。
    小坐忽惊衣袂薄，方知梧叶满中庭。

    注：诗中通过“删繁留瘦骨”、“澄澈褪虚形”等意象，展现秋日山水由丰腴转向清瘦的视觉变化。尾联以衣袂惊薄、梧叶满庭的细节，将季节流转的感知融入日常瞬间，形成物候与心境的微妙共振。全诗避免直抒胸臆，借由物象的凝练重组，传递
    出东方美学中“观物见性”的秋思传统。
    《秋窗》
    推窗忽见一庭秋，梧叶纷飞下小楼。
    几处寒砧敲冷月，谁家玉笛落孤舟。
    千山褪色风初紧，万木垂珠露未收。
    莫道霜天无客雁，长空字字写清愁。
    《秋窗》
    西风先到水晶帘，偷换梧桐一树缣。
    半壁虫声焚缥帙，满阶日影泊空檐。
    涉江人远青峰叠，抱叶蝉疏白露尖。
    唯有寒山不辞醉，斜抛红雨入诗奁。

    注：诗中通过“西风”、“梧桐”、“虫声”、“白露”等意象勾勒出深秋的静谧与清寒。尾联以拟人手法写寒山醉抛红雨，将飘零红叶喻为落入诗奁的珍宝，既点出秋色之绚烂，又暗含诗人对自然诗意的珍藏。全篇以物候变迁为经纬，织就一幅朦胧而
    精致的秋日画卷。

    --- 开始异步调用测试 ---
    异步请求 1 已发出...
    异步请求 2 已发出...
    异步请求 3 已发出...
    异步请求 3 响应返回
    异步请求 1 响应返回
    异步请求 2 响应返回
    异步总耗时: 6.44 秒
    《秋窗》
    推窗忽见一庭秋，梧叶半黄蝉半收。
    风起欲沾云气冷，雨余贪看晚山幽。
    光阴暗度杯中影，世事轻随水上鸥。
    幸有篱边数枝菊，年年岁岁伴人留。

    注：本诗通过“梧叶半黄”、“云气冷”、“晚山幽”等意象勾勒出秋日的清寂轮廓，后以杯中影、水上鸥隐喻时光流转与世事飘忽。尾联笔锋轻转，借篱边菊点出秋日亦存温厚底色——草木凋零中仍有生命坚守，传递出对自然循环的静观与接纳。
    《秋窗》
    推窗忽见一庭秋，梧叶翻黄欲下楼。
    几处寒砧敲夜月，谁家玉笛落江舟。
    千山褪色云留影，万壑收声溪不流。
    莫道西风催客老，菊香扶梦过篱头。

    注：诗中通过“梧叶翻黄”、“寒砧敲月”等意象勾勒出深秋画卷，尾联以菊香扶梦的灵动收束，在萧瑟中透出温暖生机。全诗运用通感手法，将视觉的褪色、听觉的收声、嗅觉的菊香交织成多维秋境，最后以超越物理界限的“扶梦过篱头”完成对秋 
    的精神突围。                                                                                                                                                                                                           《秋窗》
    西风先到水晶帘，偷换梧桐一树缣。
    忽有乱虫争絮语，斜阳坐在画楼檐。

    注：我的创作思路是通过细微物象的变迁捕捉秋意。以“西风”、“梧桐”勾勒季节流转的底色，“乱虫絮语”与“斜阳画檐”形成声光交织的黄昏剧场。诗中“偷换”、“争”、“坐”等动词赋予静物动态生命，试图在古典意象中注入现代诗性的凝视，让秋 
    日不再仅是萧瑟的代名词，而是充满张力与私语的灵性空间。                                                                                                                                                                 
    """