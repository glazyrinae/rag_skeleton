"""
Абстрактный интерфейс для любого хранилища индексов.
Все конкретные реализации должны наследовать этот класс.
"""

from abc import ABC, abstractmethod
from typing import List, Any, Optional
from llama_index.core import StorageContext


class StorageBackend(ABC):
    """Контракт для управления индексами, метаданными и векторами."""

    @abstractmethod
    def get_vector_store(self) -> Any:
        """Возвращает инициализированное векторное хранилище."""
        pass

    @abstractmethod
    def build_context(self, index_type: str) -> StorageContext:
        """Создаёт и возвращает StorageContext для указанного типа индекса."""
        pass

    @abstractmethod
    def persist_context(self, ctx: StorageContext, index_type: str) -> None:
        """Сохраняет состояние docstore и index_store."""
        pass

    @abstractmethod
    def index_exists(self, index_type: str) -> bool:
        """Проверяет, существует ли уже сохранённый индекс."""
        pass

    @abstractmethod
    def reset_index(self, index_type: str) -> None:
        """Полностью удаляет/сбрасывает индекс и его метаданные."""
        pass

    @abstractmethod
    def save_bm25_nodes(self, nodes: List[Any], db_name: str) -> None:
        """Сохраняет узлы для BM25-ретривера."""
        pass

    @abstractmethod
    def load_bm25_nodes(self, db_name: str) -> Optional[List[Any]]:
        """Загружает узлы для BM25-ретривера."""
        pass
