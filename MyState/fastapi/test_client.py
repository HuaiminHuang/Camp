def test():
    import requests

    # 测试根路径
    response = requests.get("http://127.0.0.1:8000/")
    print("根路径响应:", response.json())

    # 测试items路径，带路径参数和查询参数
    response = requests.get("http://127.0.0.1:8000/items/42", params={"q": "test"})
    print("Items路径响应:", response.json())

    # 测试不带查询参数的items路径
    response = requests.get("http://127.0.0.1:8000/items/99")
    print("Items路径响应(无查询参数):", response.json())

if __name__ == "__main__":
    test()
    