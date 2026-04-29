"""
Реализация хранилища на PostgreSQL.
Векторы хранятся через pgvector, метадаты и BM25 — в JSONB/BYTEA колонках.
"""

import json
import pickle
from urllib.parse import urlparse, unquote
from typing import List, Any, Optional
from contextlib import contextmanager
import psycopg2
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.postgres import PGVectorStore
from .storage_backend import StorageBackend

# Схема БД (выполните один раз при первом запуске):
"""
CREATE TABLE IF NOT EXISTS rag_storage (
    id SERIAL PRIMARY KEY,
    db_name TEXT NOT NULL,
    index_type TEXT NOT NULL,
    component TEXT NOT NULL, -- 'docstore', 'indexstore', 'bm25'
    data_jsonb JSONB,
    data_bytes BYTEA,
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(db_name, index_type, component)
);
CREATE INDEX idx_rag_storage_lookup ON rag_storage(db_name, index_type, component);
"""


class PostgresBackend(StorageBackend):
    def __init__(self, db_name: str, connection_string: str, embed_dim: int = 384):
        self.db_name = db_name
        self.conn_str = connection_string
        self.embed_dim = embed_dim
        parsed = urlparse(self.conn_str)
        scheme = parsed.scheme or "postgresql"
        host = parsed.hostname
        port = parsed.port
        database = parsed.path.lstrip("/") if parsed.path else None
        user = unquote(parsed.username) if parsed.username else None
        password = unquote(parsed.password) if parsed.password else None
        async_conn_str = (
            f"{scheme}+asyncpg://{user}:{password}@{host}:{port}/{database}"
            if all([host, port, database, user, password])
            else None
        )

        # Инициализируется ТОЛЬКО если выбран этот бэкенд
        self.vector_store = PGVectorStore.from_params(
            connection_string=self.conn_str,
            async_connection_string=async_conn_str,
            host=host,
            port=str(port) if port is not None else None,
            database=database,
            user=user,
            password=password,
            table_name=f"pgvector_{db_name}",
            embed_dim=embed_dim,
            hybrid_search=False,
            text_search_config="english",
        )

    @contextmanager
    def _get_conn(self):
        conn = psycopg2.connect(self.conn_str)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def get_vector_store(self) -> PGVectorStore:
        return self.vector_store

    def build_context(self, index_type: str) -> StorageContext:
        docstore = self._load_metadata(index_type, "docstore")
        index_store = self._load_metadata(index_type, "indexstore")

        return StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=docstore,
            index_store=index_store,
        )

    def persist_context(self, ctx: StorageContext, index_type: str) -> None:
        doc_dict = ctx.docstore.to_dict()
        idx_dict = ctx.index_store.to_dict()

        self._save_metadata(index_type, "docstore", doc_dict)
        self._save_metadata(index_type, "indexstore", idx_dict)

    def index_exists(self, index_type: str) -> bool:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT COUNT(*) FROM rag_storage WHERE db_name = %s AND index_type = %s AND component = 'indexstore'",
                    (self.db_name, index_type),
                )
                return cur.fetchone()[0] > 0

    def reset_index(self, index_type: str) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "DELETE FROM rag_storage WHERE db_name = %s AND index_type = %s",
                    (self.db_name, index_type),
                )
                # Очистка векторов (опционально, зависит от логики приложения)
                # cur.execute(f"DELETE FROM pgvector_{self.db_name} WHERE ...")

    def save_bm25_nodes(self, nodes: List[Any], db_name: str) -> None:
        data = pickle.dumps(nodes, protocol=pickle.HIGHEST_PROTOCOL)
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_storage (db_name, index_type, component, data_bytes, updated_at)
                    VALUES (%s, 'bm25', 'bm25', %s, NOW())
                    ON CONFLICT (db_name, index_type, component)
                    DO UPDATE SET data_bytes = EXCLUDED.data_bytes, updated_at = NOW()
                    """,
                    (self.db_name, psycopg2.Binary(data)),
                )

    def load_bm25_nodes(self, db_name: str) -> Optional[List[Any]]:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data_bytes FROM rag_storage WHERE db_name = %s AND index_type = 'bm25' AND component = 'bm25'",
                    (self.db_name,),
                )
                row = cur.fetchone()
                if row and row[0]:
                    return pickle.loads(bytes(row[0]))
                return None

    def _load_metadata(self, index_type: str, component: str) -> Any:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT data_jsonb FROM rag_storage WHERE db_name = %s AND index_type = %s AND component = %s",
                    (self.db_name, index_type, component),
                )
                row = cur.fetchone()
                if row and row[0]:
                    if component == "docstore":
                        return SimpleDocumentStore.from_dict(row[0])
                    elif component == "indexstore":
                        return SimpleIndexStore.from_dict(row[0])
        return SimpleDocumentStore() if component == "docstore" else SimpleIndexStore()

    def _save_metadata(self, index_type: str, component: str, data: dict) -> None:
        with self._get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    INSERT INTO rag_storage (db_name, index_type, component, data_jsonb, updated_at)
                    VALUES (%s, %s, %s, %s, NOW())
                    ON CONFLICT (db_name, index_type, component)
                    DO UPDATE SET data_jsonb = EXCLUDED.data_jsonb, updated_at = NOW()
                    """,
                    (
                        self.db_name,
                        index_type,
                        component,
                        json.dumps(data, ensure_ascii=False, default=str),
                    ),
                )
