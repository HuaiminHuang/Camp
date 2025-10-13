import asyncio
import json
import os
import sys
from typing import Optional
from contextlib import AsyncExitStack

from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from openai import AsyncOpenAI

# 加载 .env 环境变量（建议里面包含 OPENAI_API_KEY）
load_dotenv()


class MCPClient:
    """MCP + OpenAI 工具增强客户端"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()
        self.openai = AsyncOpenAI(base_url='https://api.openai-proxy.org/v1',
                                   api_key=os.getenv("OPENAI_API_KEY"))

    async def connect_to_server(self, server_script_path: str):
        """连接到本地 MCP 服务器"""

        is_python = server_script_path.endswith(".py")
        is_js = server_script_path.endswith(".js")
        if not (is_python or is_js):
            raise ValueError("Server script must be a .py or .js file")

        command = "python" if is_python else "node"
        server_params = StdioServerParameters(
            command=command,
            args=[server_script_path],
            env=None,
        )

        print(f"🚀 启动 MCP 服务器：{command} {server_script_path}")

        # 建立 stdio 通道
        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport

        # 建立会话
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))
        await self.session.initialize()

        # 列出服务器提供的工具
        response = await self.session.list_tools()
        tools = response.tools
        print("\n✅ 已连接到 MCP 服务器，工具列表：", [tool.name for tool in tools])

    async def process_query(self, query: str) -> str:
        """处理自然语言查询：使用 GPT + MCP 工具"""

        messages = [{"role": "user", "content": query}]

        # 从 MCP 获取可用工具
        response = await self.session.list_tools()
        available_tools = response.tools

        tools = [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description or "",
                    "parameters": t.inputSchema,
                },
            }
            for t in available_tools
        ]

        # 第一次 GPT 调用
        gpt_response = await self.openai.chat.completions.create(
            model="gpt-4.1",  # 或 "gpt-5"（企业版）
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )

        message = gpt_response.choices[0].message
        final_text = []

        # 若模型决定调用工具
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    print(f"⚠️ 工具参数解析失败: {e}")
                    tool_args = {}

                print(f"\n🧩 调用工具：{tool_name} 参数：{tool_args}")

                # 执行 MCP 工具
                tool_result = await self.session.call_tool(tool_name, tool_args)
                result_text = tool_result.content[0].text if tool_result.content else "No content returned"

                print(f"✅ 工具返回：{result_text}")


                # 把工具结果传回 GPT
                messages.append(
                    {
                        "role": "assistant",
                        "tool_calls": message.tool_calls,
                    }
                )
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result_text,
                    }
                )

            # 第二轮 GPT 推理（带工具结果）
            final_completion = await self.openai.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
            )
            return final_completion.choices[0].message.content

        else:
            # 没有调用工具，直接返回回答
            return message.content

    async def chat_loop(self):
        """交互式聊天循环"""
        print("\n💬 MCP Client Started!")
        print("输入问题开始对话，输入 'quit' 退出。\n")

        while True:
            try:
                query = input("\nQuery: ").strip()
                if query.lower() in {"quit", "exit"}:
                    break

                response = await self.process_query(query)
                print("\n🤖 GPT:", response)

            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")

    async def cleanup(self):
        """关闭连接"""
        await self.exit_stack.aclose()


async def main():
    if len(sys.argv) < 2:
        print("用法: python mcp_openai_client.py <path_to_server_script>")
        sys.exit(1)

    client = MCPClient()
    try:
        await client.connect_to_server(sys.argv[1])
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
