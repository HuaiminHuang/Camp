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
        self.history = []
        self.max_history = 20  # 最多保留最近 20 条消息

    async def process_query(self, query: str) -> str:
        """处理自然语言查询：GPT + MCP 工具 + 历史 + roll out + 调试信息"""

        # 1️⃣ 保存用户消息到历史
        self.history.append({"role": "user", "content": query})

        # 2️⃣ roll out: 保留最近 max_history 条
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        # 3️⃣ 准备 messages（只保留 user + assistant）
        messages = [m for m in self.history if m['role'] in ("user", "assistant")]
        if not messages:
            messages = [{"role": "user", "content": query}]

        print("📝 调试: 当前 GPT 消息输入：")
        for i, m in enumerate(messages):
            print(f"  {i} | {m['role']}: {m.get('content', '')}")

        # 4️⃣ 获取可用工具
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

        print(f"🛠 可用工具: {[t['function']['name'] for t in tools]}")

        # ------------------------
        # 5️⃣ 第一次 GPT 调用
        # ------------------------
        gpt_response = await self.openai.chat.completions.create(
            model="gpt-4.1",
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=1000,
        )

        message = gpt_response.choices[0].message
        final_text = message.content or ""

        # ------------------------
        # 6️⃣ 工具调用逻辑
        # ------------------------
        if hasattr(message, "tool_calls") and message.tool_calls:
            for tool_call in message.tool_calls:
                tool_name = tool_call.function.name
                try:
                    tool_args = json.loads(tool_call.function.arguments)
                except Exception as e:
                    print(f"⚠️ 工具参数解析失败: {e}")
                    tool_args = {}

                print(f"\n🧩 调用工具: {tool_name}")
                print(f"   参数: {tool_args}")

                # 执行 MCP 工具
                tool_result = await self.session.call_tool(tool_name, tool_args)
                result_text = ""
                if tool_result.content and isinstance(tool_result.content, list):
                    result_text = tool_result.content[0].text or ""

                print(f"✅ 工具返回: {result_text}")

                # ------------------------
                # 6a️⃣ 把 assistant + tool 返回传给 GPT
                # ------------------------
                messages.append({
                    "role": "assistant",
                    "tool_calls": [tool_call]  # 每次只对应当前 tool_call
                })
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result_text
                })

            # ------------------------
            # 6b️⃣ 第二轮 GPT 推理（带工具结果）
            # ------------------------
            final_completion = await self.openai.chat.completions.create(
                model="gpt-4.1",
                messages=messages,
            )
            final_text = final_completion.choices[0].message.content or ""

        # ------------------------
        # 7️⃣ 保存助手回复到历史，并 roll out
        # ------------------------
        self.history.append({"role": "assistant", "content": final_text})
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        print(f"\n🤖 GPT 最终回答: {final_text}\n")
        print(f"📝 当前历史长度: {len(self.history)} 条")

        return final_text



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
