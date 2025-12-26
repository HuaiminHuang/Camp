from fastapi import FastAPI

app = FastAPI()

fake_items_db = [{"item_name": "Foo"}, {"item_name": "Bar"}, {"item_name": "Baz"}]

# When you declare other function parameters that are not part of the path parameters, 
# they are automatically interpreted as "query" parameters.
@app.get("/items/")
async def read_item(skip: int = 0, limit: int = 10):
    return fake_items_db[skip : skip + limit]

# 示例输出
"""
http://127.0.0.1:8000/items/?skip=0&limit=2
return 
    [
    {
        "item_name": "Foo"
    },
    {
        "item_name": "Bar"
    },
    ]
"""

@app.get("/items/{item_id}")
async def read_items(item_id: str, q: str | None = None, short: bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item

"""
http://127.0.0.1:8000/items/foo?short=False
return
    {
    "item_id": "foo",
    "description": "This is an amazing item that has a long description"
    }
"""

# Multiple path and query parameters
# to make a query parameter required, you can just not declare any default value:
@app.get("/users/{user_id}/items/{item_id}")
async def read_user_item(
    user_id: int, item_id: str, q: str, short: bool = False
):
    item = {"item_id": item_id, "owner_id": user_id}
    if q:
        item.update({"q": q})
    if not short:
        item.update(
            {"description": "This is an amazing item that has a long description"}
        )
    return item

"""
http://127.0.0.1:8000/users/114514/items/42?q=universal&short=false
return
    {
        "item_id": "42",
        "owner_id": 114514,
        "q": "universal",
        "description": "This is an amazing item that has a long description"
    }

如果缺少没有初始值(required value)
{
  "detail": [
    {
      "type": "missing",
      "loc": [
        "query",
        "q"
      ],
      "msg": "Field required",
      "input": null
    }
  ]
}
"""