"""
FastAPI CRUD 操作示例
实现一个简单的商品管理系统，包含创建、读取、更新、删除操作
"""

from fastapi import FastAPI
from pydantic import BaseModel

# 数据模型定义
class Item(BaseModel):
    """
    商品数据模型
    
    字段说明：
    - name: 商品名称（必填）
    - description: 商品描述（可选）
    - price: 商品价格（必填，浮点数）
    - tax: 税费（可选，浮点数）
    """
    name : str
    description: str | None = None
    price: float
    tax: float | None = None

# 创建 FastAPI 应用实例
app = FastAPI()

# 内存中的商品列表（实际项目中应使用数据库）
items_list = []

# CRUD 

@app.post("/items/")
async def create_item(item: Item):
    """
    创建新商品
    功能：接收商品数据并添加到列表中
    请求示例：
    curl -X POST "http://127.0.0.1:8000/items/" \
         -H "Content-Type: application/json" \
         -d '{"name": "laptop", "description": "Gaming laptop", "price": 999.99, "tax": 50}'
    
    响应：返回创建的商品对象
    """
    items_list.append(item)
    return item

@app.get("/items/")
async def get_items():
    """
    获取所有商品列表
    功能：返回内存中所有商品的列表
    请求示例：
    curl "http://127.0.0.1:8000/items/"
    响应：返回商品列表数组
    """
    return items_list

@app.get("/items/{item_name}")
async def get_item(item_name: str):
    """
    根据名称获取特定商品
    功能：在商品列表中查找指定名称的商品
    参数：
    - item_name: 要查找的商品名称
    请求示例：
    curl "http://127.0.0.1:8000/items/laptop"
    响应：返回找到的商品或错误信息
    """
    for item in items_list:
        if item.name == item_name:
            return item
    return {"error": "Item not found"}

@app.put("/items/{item_name}")
async def update_item(item_name: str, item: Item):
    """
    更新指定商品信息
    功能：根据名称查找商品并用新数据替换
    参数：
    - item_name: 要更新的商品名称
    - item: 新的商品数据
    请求示例：
    curl -X PUT "http://127.0.0.1:8000/items/laptop" \
         -H "Content-Type: application/json" \
         -d '{"name": "laptop", "description": "Updated gaming laptop", "price": 899.99, "tax": 45}'
    响应：返回更新成功信息和新商品数据
    """
    for i, existing_item in enumerate(items_list):
        if existing_item.name == item_name:
            items_list[i] = item
            return {"message": "Item updated successfully", "item": item}
    return {"error": "Item not found"}

@app.delete("/items/{item_name}")
async def delete_item(item_name: str):
    """
    删除指定商品
    功能：根据名称查找并删除商品
    参数：
    - item_name: 要删除的商品名称
    请求示例：
    curl -X DELETE "http://127.0.0.1:8000/items/laptop"
    响应：返回删除成功信息和被删除的商品数据
    """
    for i, item in enumerate(items_list):
        if item.name == item_name:
            deleted_item = items_list.pop(i)
            return {"message": "Item deleted successfully", "item": deleted_item}
    return {"error": "Item not found"}
