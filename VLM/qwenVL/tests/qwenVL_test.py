from openai import OpenAI
from dotenv import load_dotenv
import os

# pip install dashscope
from dashscope import MultiModalConversation
import dashscope 
import pprint

# set your DASHSCOPE_API_KEY here
load_dotenv()
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY")

def stream_print(response):
    # 注意：这里的 response 是一个生成器对象
    L=0
    for chunk in response:
        # if L<1:
        #     print("\n--- 这是一个 Chunk 的原始结构 ---")
        #     pprint.pprint(chunk) 
        #     L+=1
        #     print("-------------------------------\n")
        if chunk.status_code == 200:
            # 这里的 chunk 才是包含 output 属性的对象
            # 对于 Qwen-VL，内容通常在 content[0]['text']
            content = chunk.output.choices[0].message.content
            if isinstance(content, list) and len(content) > 0:
                # 获取当前块的文本内容
                text = content[0].get('text', '')
                print(text, end="", flush=True)
        else:
            print(f"\n错误: {chunk.message}")

def base_url_test():
    client = OpenAI(
        api_key=DASHSCOPE_API_KEY,
        base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    completion = client.chat.completions.create(
        model="qwen3-vl-235b-a22b-instruct",
        messages=[{"role": "user", "content": [
            {"type": "image_url",
            "image_url": {"url": "https://dashscope.oss-cn-beijing.aliyuncs.com/images/dog_and_girl.jpeg"}},
            {"type": "text", "text": "这是什么"},
        ]}]
    )
    print(completion.model_dump_json())
    # print(completion.choices[0].message.content)

def local_img():
    # 若使用新加坡地域的模型，请取消下列注释
    # dashscope.base_http_api_url = "https://dashscope-intl.aliyuncs.com/api/v1"

    # 替换为你本地图像的绝对路径
    local_path = os.path.join(os.path.dirname(__file__), "test_file", "MHWs.jpg")
    image_path = f"file://{local_path}"
    messages = [
                    {'role':'user',
                    'content': [{'image': image_path},
                                {'text': '请你详细讲解图片中的内容'}]}]
    if not DASHSCOPE_API_KEY:
        raise ValueError("need api key to call model")
    
    response = MultiModalConversation.call(
        api_key=DASHSCOPE_API_KEY,
        model='qwen3-vl-235b-a22b-instruct',  
        messages=messages,
        stream=True,
        incremental_output=True,
        )
    
    # print(response)
    # print(response.output.choices[0].message.content[0]["text"])
    stream_print(response)

def local_video():
    local_path = os.path.join(os.path.dirname(__file__), "test_file", "Monster_Hunter_Wilds_the_fourth_updating.mp4")
    video_path = f"file://{local_path}"
    messages = [
                    {'role':'user',
                    # fps参数控制视频抽帧数量，表示每隔1/fps 秒抽取一帧
                    'content': [{'video': video_path,"fps":2},
                                {'text': '这段视频描绘的是什么内容？'}]}]
    response = MultiModalConversation.call(
        # 若没有配置环境变量，请用百炼API Key将下行替换为：api_key="sk-xxx"
        api_key=DASHSCOPE_API_KEY,
        model='qwen3-vl-235b-a22b-instruct',  
        messages=messages,
        stream=True,
        incremental_output=True,
        )
    # print(response.output.choices[0].message.content[0]["text"])
    stream_print(response)

if __name__ == "__main__":
    local_video()
    # local_img()
    # base_url_test()