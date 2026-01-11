import os
from dotenv import load_dotenv
import requests 
import json
from langchain_tavily import TavilySearch
import datetime
from openai import OpenAI
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Tuple

load_dotenv("apikey.env")

api_key = os.environ.get("OPENAI_API_KEY")
if not api_key:
    raise ValueError("Error: OPENAI_API_KEY environment variable not set")

# 注意：base_url 末尾不要有多余空格！
client = OpenAI(api_key=api_key, base_url="https://api.openai-proxy.org/v1")


# 定义 ReAct 步骤的结构化输出模型
class ReActStep(BaseModel):
    """
    LLM 每一步 ReAct 推理的结构化输出。
    保留原始 ReAct 语义：Thought / Action / Observation / Final Answer
    用 JSON 字段表达
    """
    thought: str = Field(..., description="当前推理思考，必须包含")
    action: Optional[str] = Field(None, description="要调用的工具名称，如 'tavily_search_results_json'")
    action_input: Optional[Dict[str, Any]] = Field(None, description="工具调用的参数字典")
    final_answer: Optional[str] = Field(None, description="如果可以直接回答，则填写最终答案")

# 工具初始化（Tavily）
tavily_api_key = os.getenv("TAVILY-API-KEY")
if tavily_api_key:
    os.environ["TAVILY_API_KEY"] = tavily_api_key
else:
    print("警告: 未设置 TAVILY-API-KEY 环境变量.")

tavily = TavilySearch(max_results=4)
tools = [tavily]
tool_names = " or ".join([tool.name for tool in tools])


# 构建工具描述（用于 prompt）
tool_descs = ""
for tool in tools:
    desc = f"- {tool.name}: {tool.description}\n"
    if hasattr(tool, 'args') and tool.args:
        desc += "  参数:\n"
        for arg_name, arg_info in tool.args.items():
            desc += f"    - {arg_name} ({arg_info.get('type', 'str')}): {arg_info.get('description', '')}\n"
    tool_descs += desc


# 结构化 LLM 调用函数（强制 JSON 输出）
def llm_structured(prompt: str) -> ReActStep:
    """
    调用 LLM 并强制其返回符合 ReActStep 的 JSON。
    """
    messages = [{"role": "user", "content": prompt}]
    try:
        resp = client.chat.completions.create(
            model="gpt-4.1",  
            messages=messages,
            response_format={"type": "json_object"},  # 👈 关键：强制 JSON 输出
            temperature=0.1,
            max_tokens=1024,
            timeout=30
        )
        raw_content = resp.choices[0].message.content
        return ReActStep.model_validate_json(raw_content)
    except Exception as e:
        raise RuntimeError(f"LLM 结构化调用失败: {e}")


# 5. ReAct Agent 执行函数（核心重写）
def agent_execute(query: str, chat_history: List[Tuple[str, str]] = None) -> Tuple[bool, str, List[Tuple[str, str]]]:
    """
    执行 ReAct Agent
    
    输入:
        query: 用户当前问题
        chat_history: 历史对话 [(问, 答), ...]
    
    输出:
        (success: bool, result: str, updated_chat_history: list)
    
    核心逻辑:
        1. 构建包含历史和 scratchpad 的 prompt
        2. 调用 LLM 获取结构化 ReActStep
        3. 若有 final_answer → 返回
        4. 若有 action → 调用工具，拼接 Observation 到 scratchpad
        5. 循环直到得到最终答案或失败
    """
    if chat_history is None:
        chat_history = []

    # agent_scratchpad 仍以原始 ReAct 文本格式记录，供下一轮 LLM 阅读
    agent_scratchpad = ""

    # 最大工具调用次数防死循环
    max_iterations = 5
    iteration = 0

    while iteration < max_iterations:
        iteration += 1

        # 构建历史对话文本（用于 prompt）
        history_text = "\n".join([f"Question: {q}\nAnswer: {a}" for q, a in chat_history])
        today = datetime.datetime.now().strftime('%Y-%m-%d')

        # 构建 ReAct Prompt（要求 LLM 输出 JSON，但展示历史时用原始文本格式）
        prompt = f"""
Today is {today}. You are a helpful ReAct agent with access to external tools.

Available tools:
{tool_descs}

Previous conversation:
{history_text}

Current task:
Question: {query}

Previous ReAct steps (if any):
{agent_scratchpad}

Now, think step by step. Respond ONLY in valid JSON with the following keys:
- "thought": your reasoning (required)
- "action": tool name to call, or null if you can answer directly
- "action_input": {{...}} parameters for the tool, or null
- "final_answer": your final answer, or null if you need to use a tool

Rules:
- If you know the answer, set "final_answer" and leave action/action_input as null.
- If you need to search, set "action" and "action_input", leave "final_answer" as null.
- Output ONLY JSON, no markdown, no explanation.

Begin:
        """.strip()

        print('\033[32m---等待LLM返回... ...\n%s\n\033[0m' % prompt, flush=True)

        # 调用结构化 LLM
        try:
            step: ReActStep = llm_structured(prompt)
        except Exception as e:
            return False, f"LLM 调用异常: {e}", chat_history

        print('\033[34m---LLM 返回（结构化）---\nThought: %s\nAction: %s\nAction Input: %s\nFinal Answer: %s\n---\033[0m' %
              (step.thought, step.action, step.action_input, step.final_answer), flush=True)

        # 能直接回答/ 完成回答
        if step.final_answer is not None:
            chat_history.append((query, step.final_answer))
            return True, step.final_answer, chat_history

        # 需要调用工具
        if step.action is not None and step.action_input is not None:
            # 查找匹配的工具
            the_tool = None
            for t in tools:
                if t.name == step.action:
                    the_tool = t
                    break

            if the_tool is None:
                observation = f"Error: Tool '{step.action}' not found."
            else:
                try:
                    # 使用tools
                    tool_result = the_tool.invoke(step.action_input)
                    observation = str(tool_result)
                except Exception as e:
                    observation = f"Tool execution error: {e}"

            # 将本次 ReAct 步骤以原始文本格式加到 scratchpad
            agent_scratchpad += (
                f"Thought: {step.thought}\n"
                f"Action: {step.action}\n"
                f"Action Input: {json.dumps(step.action_input, ensure_ascii=False)}\n"
                f"Observation: {observation}\n"
            )

        else:
            # LLM 返回了无效状态（既无 final_answer，又无完整 action）
            return False, "LLM 返回无效 ReAct 步骤：缺少 action 或 final_answer", chat_history

    # 超出最大迭代次数
    return False, "Agent 超过最大工具调用次数（5次），未能得出答案。", chat_history



# 带重试的执行函数
def agent_execute_with_retry(query: str, chat_history: List[Tuple[str, str]] = None, retry_times: int = 3):
    if chat_history is None:
        chat_history = []
    for i in range(retry_times):
        success, result, updated_history = agent_execute(query, chat_history=chat_history)
        if success:
            return success, result, updated_history
    return success, result, updated_history


if __name__ == "__main__":
    my_history = []
    while True:
        query = input('query (输入 "q" 退出): ')
        if query.strip().lower() == "q":
            break
        success, result, my_history = agent_execute_with_retry(query, chat_history=my_history)
        my_history = my_history[-10:]  # 保留最近10轮
        if success:
            print(f"✅ Answer: {result}\033[0m\n")
        else:
            print(f"❌ Error: {result}\033[0m\n")