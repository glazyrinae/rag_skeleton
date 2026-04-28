"""
Фабрика, которая выбирает и инициализирует ТОЛЬКО нужный бэкенд.
Глубокая интеграция с конкретным хранилищем отсутствует.
"""

from typing import Optional
from .storage_backend import StorageBackend
from .local_deeplake_backend import LocalDeepLakeBackend
from .postgres_backend import PostgresBackend


class StorageManager:
    def __init__(
        self,
        db_name: str,
        backend_type: str = "local",
        base_path: Optional[str] = None,
        connection_string: Optional[str] = None,
        embed_dim: int = 384,
    ):
        self.db_name = db_name
        self.backend_type = backend_type

        if backend_type == "local":
            self.backend: StorageBackend = LocalDeepLakeBackend(
                db_name=db_name,
                base_path=base_path or "/app/deeplake_data",
            )
        elif backend_type == "postgres":
            if not connection_string:
                raise ValueError(
                    "Для бэкенда 'postgres' обязателен параметр connection_string"
                )
            self.backend = PostgresBackend(
                db_name=db_name,
                connection_string=connection_string,
                embed_dim=embed_dim,
            )
        else:
            raise ValueError(
                f"Неподдерживаемый backend_type: {backend_type}. Используйте 'local' или 'postgres'."
            )

    # Прокси-методы для удобства вызова из IndexRegistry
    def get_vector_store(self):
        return self.backend.get_vector_store()

    def build_context(self, index_type: str):
        return self.backend.build_context(index_type)

    def persist_context(self, ctx, index_type: str):
        return self.backend.persist_context(ctx, index_type)

    def index_exists(self, index_type: str) -> bool:
        return self.backend.index_exists(index_type)

    def reset_index(self, index_type: str) -> None:
        return self.backend.reset_index(index_type)

    def save_bm25_nodes(self, nodes: list) -> None:
        return self.backend.save_bm25_nodes(nodes, self.db_name)

    def load_bm25_nodes(self) -> list:
        return self.backend.load_bm25_nodes(self.db_name)
