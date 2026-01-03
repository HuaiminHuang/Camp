import os
import asyncio
import inspect
import logging
import logging.config
import numpy as np
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug

from dotenv import load_dotenv

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"
LOCAL_BOOKS = "./books/zhenhuanzhuan.txt"


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


async def print_stream(stream):
    async for chunk in stream:
        if chunk:
            print(chunk, end="", flush=True)


async def initialize_rag():
    # 原始 embedding 函数
    original_embed_func = partial(
        ollama_embed.func,
        embed_model=os.getenv("EMBEDDING_MODEL", "bge-m3:latest"),
        host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
    )

    # 包装函数，添加 NaN 检测，最大重试3次
    async def safe_embed_func(texts):
        max_retries = 3
        for attempt in range(max_retries):
            result = await original_embed_func(texts)
            if not np.isnan(result).any():
                return result
            print(f"⚠️ NaN detected in embedding, retrying... (attempt {attempt + 1}/{max_retries})")
            if attempt < max_retries - 1:
                await asyncio.sleep(0.5)
        raise ValueError("Embedding contains NaN after 3 retries")

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
            func=safe_embed_func,
        ),
    )

    await rag.initialize_storages()  # Auto-initializes pipeline_status
    return rag



async def main():
    """
    测试在不同模式下的检索结果["naive", "local", "global", "hybird"]
    """
    rag = None
    try:
        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        if rag.embedding_func is None:
            raise ValueError("embedding_func is not set")
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        # with open(LOCAL_BOOKS, "r", encoding="utf-8") as f:
        #     await rag.ainsert(f.read())

        # Perform naive search
        print("\n=====================")
        print("Query mode: naive")
        print("=====================")
        resp = await rag.aquery(
            "可以为我介绍这个故事的主题吗？",
            param=QueryParam(mode="naive", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform local search
        print("\n=====================")
        print("Query mode: local")
        print("=====================")
        resp = await rag.aquery(
            "可以为我介绍这个故事的主题吗？",
            param=QueryParam(mode="local", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform global search
        print("\n=====================")
        print("Query mode: global")
        print("=====================")
        resp = await rag.aquery(
            "可以为我介绍这个故事的主题吗？",
            param=QueryParam(mode="global", stream=True),
        )
        if inspect.isasyncgen(resp):
            await print_stream(resp)
        else:
            print(resp)

        # Perform hybrid search
        print("\n=====================")
        print("Query mode: hybrid")
        print("=====================")
        resp = await rag.aquery(
            "可以为我介绍这个故事的主题吗？",
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