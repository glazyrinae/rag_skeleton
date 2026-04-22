from functools import lru_cache
import os

from rag.rag_service import RAGService


@lru_cache(maxsize=32)
def get_rag_by_db(db_name: str, read_only: bool = False) -> RAGService:
    return RAGService(db_name=db_name.strip(), read_only=read_only)


@lru_cache()
def get_rag() -> RAGService:
    db_name = os.getenv("RAG_DB_NAME", "core")
    return get_rag_by_db(db_name, read_only=True)
