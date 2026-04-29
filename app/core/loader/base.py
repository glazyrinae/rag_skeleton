"""
Базовый контракт для любого источника документов.
"""

from abc import ABC, abstractmethod
from typing import Iterator, List
from llama_index.core import Document


class BaseDocumentLoader(ABC):
    @property
    @abstractmethod
    def source_name(self) -> str:
        pass

    @abstractmethod
    def iter_batches(self, batch_size: int = 100) -> Iterator[List[Document]]:
        pass
