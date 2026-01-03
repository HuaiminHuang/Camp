import os
import inspect
import asyncio
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache, openai_embed
from lightrag.utils import setup_logger
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc
from functools import partial

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)
setup_logger("lightrag", level="INFO")

WORKING_DIR = "./rag_storage"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)

async def llm_model_func(
    prompt, system_prompt=None, history_messages=[], keyword_extraction=False, **kwargs
) -> str:
    return await openai_complete_if_cache(
        os.getenv("LLM_MODEL", "deepseek-chat"),
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        api_key=os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY"),
        base_url=os.getenv("LLM_BINDING_HOST", "https://api.deepseek.com"),
        **kwargs,
    )

async def initialize_rag():
    rag = LightRAG(
        working_dir=WORKING_DIR,
        llm_model_func=llm_model_func,
        # Note: ollama_embed is decorated with @wrap_embedding_func_with_attrs,
        # which wraps it in an EmbeddingFunc. Using .func accesses the original
        # unwrapped function to avoid double wrapping when we create our own
        # EmbeddingFunc with custom configuration (embedding_dim, max_token_size).
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("MAX_EMBED_TOKENS", "8192")),
            func=partial(
                ollama_embed.func,  # Access the unwrapped function to avoid double EmbeddingFunc wrapping
                embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
                host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
            ),
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag

async def main():
    try:
        # Initialize RAG instance
        rag = await initialize_rag()
        docs = [
            "极限是微积分的出发点，用来精确定义变量在逼近某一数值或无穷远时的行为。",
            "导数刻画函数在某一点的瞬时变化率，是连接几何切线与实际变化规律的核心工具。",
            "积分可以理解为导数的逆运算，它通过累积无穷小量来计算面积、体积或总效应。",
            "微分与积分通过微积分基本定理紧密联系，揭示了局部变化与整体累积之间的统一性。",
            "微积分为现代科学与工程提供了分析连续系统的语言，是理解复杂动态过程的基础。"
        ]
        for t in docs:
            await rag.ainsert(t)

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        resp = await rag.aquery(
            "微积分的研究对象是什么？",
            param=QueryParam(mode="naive", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        resp = await rag.aquery(
            "微积分的研究对象是什么？",
            param=QueryParam(mode="hybrid", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()

if __name__ == "__main__":
    asyncio.run(main())