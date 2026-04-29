"""
Загрузчик документов из локальной файловой системы.
"""

from typing import Iterator, List
from llama_index.core import Document, SimpleDirectoryReader
from .base import BaseDocumentLoader


class FileLoader(BaseDocumentLoader):
    def __init__(self, path: str, recursive: bool = True, batch_size: int = 100):
        self.path = path
        self.recursive = recursive
        self._batch_size = batch_size

    @property
    def source_name(self) -> str:
        return f"file://{self.path}"

    def iter_batches(self, batch_size: int = None) -> Iterator[List[Document]]:
        bs = batch_size or self._batch_size
        reader = SimpleDirectoryReader(self.path, recursive=self.recursive)

        current_batch: List[Document] = []
        for doc in reader.iter_data():
            items = doc if isinstance(doc, list) else [doc]
            current_batch.extend(items)

            while len(current_batch) >= bs:
                yield current_batch[:bs]
                current_batch = current_batch[bs:]

        if current_batch:
            yield current_batch
