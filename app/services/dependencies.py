from functools import lru_cache
import os

from rag.rag_service import RAGService


@lru_cache()
def get_rag() -> RAGService:
    db_name = os.getenv("RAG_DB_NAME", "core")
    return RAGService(db_name=db_name)
