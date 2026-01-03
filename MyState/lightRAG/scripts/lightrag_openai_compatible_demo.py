import os
import asyncio
import inspect
import logging
import logging.config
from functools import partial
from lightrag import LightRAG, QueryParam
from lightrag.llm.openai import openai_complete_if_cache
from lightrag.llm.ollama import ollama_embed
from lightrag.utils import EmbeddingFunc, logger, set_verbose_debug

from dotenv import load_dotenv
import numpy as np

load_dotenv(dotenv_path=".env", override=False)

WORKING_DIR = "./dickens"
# LOCAL_BOOKS = "./books/saye.txt"
LOCAL_BOOKS = "./books/saye_test.txt"


def configure_logging():
    """Configure logging for the application"""

    # Reset any existing handlers to ensure clean configuration
    for logger_name in ["uvicorn", "uvicorn.access", "uvicorn.error", "lightrag"]:
        logger_instance = logging.getLogger(logger_name)
        logger_instance.handlers = []
        logger_instance.filters = []

    # Get log directory path from environment variable or use current directory
    log_dir = os.getenv("LOG_DIR", os.getcwd())
    log_file_path = os.path.abspath(
        os.path.join(log_dir, "lightrag_compatible_demo.log")
    )

    print(f"\nLightRAG compatible demo log file: {log_file_path}\n")
    os.makedirs(os.path.dirname(log_dir), exist_ok=True)

    # Get log file max size and backup count from environment variables
    log_max_bytes = int(os.getenv("LOG_MAX_BYTES", 10485760))  # Default 10MB
    log_backup_count = int(os.getenv("LOG_BACKUP_COUNT", 5))  # Default 5 backups

    logging.config.dictConfig(
        {
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(levelname)s: %(message)s",
                },
                "detailed": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "console": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stderr",
                },
                "file": {
                    "formatter": "detailed",
                    "class": "logging.handlers.RotatingFileHandler",
                    "filename": log_file_path,
                    "maxBytes": log_max_bytes,
                    "backupCount": log_backup_count,
                    "encoding": "utf-8",
                },
            },
            "loggers": {
                "lightrag": {
                    "handlers": ["console", "file"],
                    "level": "INFO",
                    "propagate": False,
                },
            },
        }
    )

    # Set the logger level to INFO
    logger.setLevel(logging.INFO)
    # Enable verbose debug if needed
    set_verbose_debug(os.getenv("VERBOSE_DEBUG", "false").lower() == "true")


if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)


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
        embed_model=os.getenv("EMBEDDING_MODEL", "qwen3-embedding:0.6B"),
        host=os.getenv("EMBEDDING_BINDING_HOST", "http://localhost:11434"),
    )

    # 包装函数，添加 NaN 检测，最大重试3次
    async def safe_embed_func(texts):
        max_retries = 3
        for attempt in range(max_retries):
            result = await original_embed_func(texts)
            if not np.isnan(result).any():
                return result
            print(f"NaN detected in embedding, retrying... (attempt {attempt + 1}/{max_retries})")
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
    # valid
    rag = None
    try:
        # # Clear old data files
        # files_to_delete = [
        #     "graph_chunk_entity_relation.graphml",
        #     "kv_store_doc_status.json",
        #     "kv_store_full_docs.json",
        #     "kv_store_text_chunks.json",
        #     "vdb_chunks.json",
        #     "vdb_entities.json",
        #     "vdb_relationships.json",
        # ]

        # for file in files_to_delete:
        #     file_path = os.path.join(WORKING_DIR, file)
        #     if os.path.exists(file_path):
        #         os.remove(file_path)
        #         print(f"Deleting old file:: {file_path}")

        # Initialize RAG instance
        rag = await initialize_rag()

        # Test embedding function
        test_text = ["This is a test string for embedding."]
        if rag.embedding_func is None:
            raise ValueError("embedding_func is invalid")
        embedding = await rag.embedding_func(test_text)
        embedding_dim = embedding.shape[1]
        print("\n=======================")
        print("Test embedding function")
        print("========================")
        print(f"Test dict: {test_text}")
        print(f"Detected embedding dimension: {embedding_dim}\n\n")

        with open(LOCAL_BOOKS, "r", encoding="utf-8") as f:
            await rag.ainsert(f.read())

        # # Perform naive search
        # print("\n=====================")
        # print("Query mode: naive")
        # print("=====================")
        # resp = await rag.aquery(
        #     "What are the top themes in this story?",
        #     param=QueryParam(mode="naive", stream=True),
        # )
        # if inspect.isasyncgen(resp):
        #     await print_stream(resp)
        # else:
        #     print(resp)

        # # Perform local search
        # print("\n=====================")
        # print("Query mode: local")
        # print("=====================")
        # resp = await rag.aquery(
        #     "What are the top themes in this story?",
        #     param=QueryParam(mode="local", stream=True),
        # )
        # if inspect.isasyncgen(resp):
        #     await print_stream(resp)
        # else:
        #     print(resp)

        # # Perform global search
        # print("\n=====================")
        # print("Query mode: global")
        # print("=====================")
        # resp = await rag.aquery(
        #     "What are the top themes in this story?",
        #     param=QueryParam(mode="global", stream=True),
        # )
        # if inspect.isasyncgen(resp):
        #     await print_stream(resp)
        # else:
        #     print(resp)

        # Perform hybrid search
        # print("\n=====================")
        # print("Query mode: hybrid")
        # print("=====================")
        # resp = await rag.aquery(
        #     "What are the top themes in this story?",
        #     param=QueryParam(mode="hybrid", stream=True),
        # )
        # if inspect.isasyncgen(resp):
        #     await print_stream(resp)
        # else:
        #     print(resp)

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        if rag:
            await rag.finalize_storages()


if __name__ == "__main__":
    # Configure logging before running the main function
    configure_logging()
    asyncio.run(main())
    print("\nDone!")
