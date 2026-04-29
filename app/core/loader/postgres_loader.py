"""
Загрузчик статей из PostgreSQL.
"""

import json
from datetime import date, datetime
from typing import Iterator, List
import psycopg2
from psycopg2.extras import RealDictCursor
from llama_index.core import Document
from .base import BaseDocumentLoader


class PostgresArticleLoader(BaseDocumentLoader):
    def __init__(
        self,
        conn_string: str,
        table: str = "articles",
        batch_size: int = 100,
        content_col: str = "content",
        metadata_cols: dict[str, str] | None = None,
    ):
        self.conn_string = conn_string
        self.table = table
        self.batch_size = batch_size
        self.content_col = content_col
        self.metadata_cols = metadata_cols or {"title": "title", "author": "author"}

    @property
    def source_name(self) -> str:
        return f"postgres://{self.table}"

    def iter_batches(self, batch_size: int = None) -> Iterator[List[Document]]:
        bs = batch_size or self.batch_size
        cols = f"{self.content_col}, " + ", ".join(self.metadata_cols.values())
        query = f"SELECT {cols} FROM {self.table} WHERE status='PB' LIMIT 1"

        with psycopg2.connect(self.conn_string) as conn:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute(query)
                batch: List[Document] = []

                for row in cur:
                    meta = {"source": "postgres"}
                    for key, col in self.metadata_cols.items():
                        meta[key] = row.get(col)

                    for k, v in meta.items():
                        if isinstance(v, str) and v.startswith(("{", "[")):
                            meta[k] = json.loads(v)
                        elif isinstance(v, (datetime, date)):
                            meta[k] = v.isoformat()

                    doc = Document(text=row[self.content_col], metadata=meta)
                    batch.append(doc)

                    if len(batch) >= bs:
                        yield batch
                        batch = []

                if batch:
                    yield batch
