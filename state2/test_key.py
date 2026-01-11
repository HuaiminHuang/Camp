
import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv("apikey.env")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(
    base_url='https://api.openai-proxy.org/v1',
    api_key=OPENAI_API_KEY,
)

chat_completion = client.chat.completions.create(
    messages=[
        {
            "role": "user",
            "content": "你好，请问你是？",
        }
    ],
    model="gpt-4.1",
)

print(chat_completion.choices[0].message)   