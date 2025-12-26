from fastapi import FastAPI

app = FastAPI()


@app.get("/items/{item_id}")
# using standard Python type annotations {item_id: int}
async def read_item(item_id: int):
    return {"item_id": item_id}

# 输出示例
"""
imput:
    curl -X 'GET' \
    'http://127.0.0.1:8000/items/MyName' \
    -H 'accept: application/json'
return
    {
        "item_id": "MyName"
    }
"""

"""
这里说明会自动解析这里的 int 3
http://127.0.0.1:8000/items/3

return
    {
        "item_id": 3
    }
==============================
类型不对的情况下出现报错
http://127.0.0.1:8000/items/foo
{
  "detail": [
    {
      "type": "int_parsing",
      "loc": [
        "path",
        "item_id"
      ],
      "msg": "Input should be a valid integer, unable to parse string as an integer",
      "input": "foo"
    }
  ]
}
"""

# 路径匹配优先级 (Path Ordering)
# 相同的路径只会展示第一个路径的内容
@app.get("/users/me")
async def read_user_me():
    return {"user_id": "the current user"}

@app.get("/users/me")
async def read_users():
    return ["Rick", "Morty"]

@app.get("/users/{user_id}")
async def read_user(user_id: str):
    return {"user_id": user_id}

# 输出示例
"""
可以看到输出的是第一个路径的内容
第二个路径的内容 ["Rick", "Morty"] 被覆盖
如果第二个输入的也是me同理

http://127.0.0.1:8000/users/me
return
    {
        "user_id": "the current user"
    }
"""

# Enum 会让参数变成一个下拉选择框。
# 用户只能从你定义的 alexnet, resnet, lenet 中选，不能乱填。
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

@app.get("/models/{model_name}")
async def get_model(model_name: ModelName):
    if model_name is ModelName.alexnet:
        return {"model_name": model_name, "message": "Deep Learning FTW!"}

    if model_name.value == "lenet":
        return {"model_name": model_name, "message": "LeCNN all the images"}

    return {"model_name": model_name, "message": "Have some residuals"}

# 示例
"""
http://127.0.0.1:8000/models/lenet
return
    {
        "model_name": "lenet",
        "message": "LeCNN all the images"
    }
"""