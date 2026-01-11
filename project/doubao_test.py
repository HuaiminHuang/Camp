def doubao_client():
    import os
    from openai import OpenAI
    # 请确保您已将 API Key 存储在环境变量 ARK_API_KEY 中
    # 初始化Openai客户端，从环境变量中读取您的API Key
    client = OpenAI(
        # 此为默认路径，您可根据业务所在地域进行配置
        base_url="https://ark.cn-beijing.volces.com/api/v3",
        # 从环境变量中获取您的 API Key
        api_key="492ae7db-d732-4619-a317-2e095040ee90",
    )

    # # Non-streaming:
    # print("----- standard request -----")
    # completion = client.chat.completions.create(
    #     # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
    #     model="doubao-1-5-thinking-pro-250415",
    #     messages=[
    #         {"role": "system", "content": "你是人工智能助手"},
    #         {"role": "user", "content": "你好"},
    #     ],
    # )
    # print(completion.choices[0].message.content)

    # Streaming:
    print("----- streaming request -----")
    stream = client.chat.completions.create(
        # 指定您创建的方舟推理接入点 ID，此处已帮您修改为您的推理接入点 ID
        model="doubao-1-5-thinking-pro-250415",
        messages=[
            {"role": "system", "content": "你是人工智能助手"},
            {"role": "user", "content": "豆包你好笨(～￣(OO)￣)ブ？"},
        ],
        # 响应内容是否流式返回
        stream=True,
    )
    for chunk in stream:
        if not chunk.choices:
            continue
        print(chunk.choices[0].delta.content, end="", flush=True)
    print()


def doubao_request():
    import os
    import json
    import requests
    import time

    # 配置部分
    BASE_URL = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    API_KEY = "492ae7db-d732-4619-a317-2e095040ee90"

    # 🔍 自动判断是否需要 Bearer
    if API_KEY.startswith("volc-"):
        AUTH_HEADER = f"Bearer {API_KEY}"  # 旧版 key
    else:
        AUTH_HEADER = API_KEY  # 新版 key，无需 Bearer

    # 请求头
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AUTH_HEADER}"
    }

    # 请求体（一个最小化对话）
    data = {
        "model": "ep-20251029235823-rpsx4",   # 请替换为你的 endpoint ID
        "messages": [
            {"role": "system", "content": "你是一个智能助手。"},
            {"role": "user", "content": "你好，请用一句话介绍你自己。"}
        ],
        "temperature": 0.7,
        "max_tokens": 200
    }

    print(f"正在请求 {BASE_URL} ...")
    t0 = time.time()

    try:
        response = requests.post(BASE_URL, headers=headers, json=data, timeout=10)
        cost = time.time() - t0

        print(f"耗时: {cost:.2f}s")
        print(f"HTTP状态码: {response.status_code}")
        print("resp.status_code:", response.status_code)
        print("resp.headers:", response.headers)
        print("resp.text:", response.text)
        if response.status_code == 200:
            print("请求成功，返回内容：")
            print(json.dumps(response.json(), ensure_ascii=False, indent=2))
        else:
            print("请求失败，返回：")
            print(response.text)

    except Exception as e:
        print("请求异常：", e)

if __name__ == "__main__":
    doubao_client()
    doubao_request()