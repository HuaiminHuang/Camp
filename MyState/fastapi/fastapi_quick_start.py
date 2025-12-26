from typing import Union
from fastapi import FastAPI
import logging
from logging.handlers import RotatingFileHandler
# 配置日志记
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(),               # 控制台
        logging.FileHandler("app.log", encoding="utf-8"),  # 文件
    ]
)

"""
启动命令:
    # 底层启动
    uvicorn fastapi_quick_start:app --reload
    # 更友好的界面
    fastapi dev fastapi_quick_start.py
"""

# Step 1-2: 导入FastAPI并创建应用实例
# FastAPI是一个类，封装了整个API框架的功能
# app是FastAPI应用对象，是整个应用的核心，所有路由、路径操作都通过它注册
app = FastAPI()


# Step 3-5: 定义路径操作装饰器和处理函数
# @app.get("/") 是路径操作装饰器，告诉FastAPI这个函数处理根路径("/")的GET请求
# read_root是路径操作函数，当对根路径发GET请求时会被调用
@app.get("/")
def read_root():
    # FastAPI自动将返回的字典转换为JSON响应
    return {"HelloWorld!": "这里是第一个FastAPI程序"}


# Step 3-5: 带路径参数的路径操作
# @app.get("/items/{item_id}") 处理路径包含参数的GET请求
# item_id是路径参数，自动转换为int类型
# q是查询参数，可选的字符串类型，默认为None
@app.get("/items/{item_id}")
def read_item(item_id: int, q: Union[str, None] = None):
    # 返回包含路径参数和查询参数的字典，FastAPI自动转换为JSON
    return {"item_id": item_id, "q": q}

