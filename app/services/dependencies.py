import os
from functools import lru_cache
from core.db import VectorDB

from core.llm import Llm


@lru_cache()
def get_llm() -> Llm:
    """Создает и кэширует единственный экземпляр RAG системы"""
    llm = Llm()
    return llm


@lru_cache()
def get_db() -> VectorDB:
    # QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
    # QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
    db = VectorDB()

    return db
