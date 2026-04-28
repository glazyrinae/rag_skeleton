"""
Реализация локального хранилища с использованием DeepLake для векторов
и JSON/Pickle файлов для метаданных и BM25.
"""

import os
import json
import pickle
import shutil
from typing import List, Any, Optional
from pathlib import Path
from llama_index.core import StorageContext
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from .storage_backend import StorageBackend


class LocalDeepLakeBackend(StorageBackend):
    def __init__(self, db_name: str, base_path: str = "/app/deeplake_data"):
        self.db_name = db_name
        self.base_path = Path(base_path)
        self.paths = {
            "docstore": self.base_path / f"{db_name}_docstore",
            "tree": self.base_path / f"{db_name}_tree_indexstore",
            "vector": self.base_path / f"{db_name}_vector_indexstore",
            "kg": self.base_path / f"{db_name}_kg_storage",
            "bm25": self.base_path / f"{db_name}_bm25_storage",
        }

        # Инициализируется ТОЛЬКО если выбран этот бэкенд
        self.vector_store = DeepLakeVectorStore(
            dataset_path=str(self.base_path / db_name),
            overwrite=False,
            read_only=False,
        )

    def get_vector_store(self) -> DeepLakeVectorStore:
        return self.vector_store

    def build_context(self, index_type: str) -> StorageContext:
        if index_type not in self.paths:
            raise ValueError(f"Неизвестный тип индекса: {index_type}")

        idx_path = self.paths[index_type]
        doc_path = self.paths["docstore"]

        doc_path.mkdir(parents=True, exist_ok=True)
        idx_path.mkdir(parents=True, exist_ok=True)

        doc_file = doc_path / "docstore.json"
        idx_file = idx_path / "index_store.json"

        docstore = (
            SimpleDocumentStore.from_persist_dir(str(doc_path))
            if doc_file.exists()
            else SimpleDocumentStore()
        )
        index_store = (
            SimpleIndexStore.from_persist_dir(str(idx_path))
            if idx_file.exists()
            else SimpleIndexStore()
        )

        return StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=docstore,
            index_store=index_store,
        )

    def persist_context(self, ctx: StorageContext, index_type: str) -> None:
        idx_path = self.paths[index_type]
        doc_path = self.paths["docstore"]
        idx_path.mkdir(parents=True, exist_ok=True)
        doc_path.mkdir(parents=True, exist_ok=True)

        ctx.docstore.persist(str(doc_path / "docstore.json"))
        ctx.index_store.persist(str(idx_path / "index_store.json"))

    def index_exists(self, index_type: str) -> bool:
        idx_file = self.paths[index_type] / "index_store.json"
        return idx_file.exists()

    def reset_index(self, index_type: str) -> None:
        if index_type not in self.paths:
            return
        path = self.paths[index_type]
        if path.exists():
            shutil.rmtree(str(path))
            path.mkdir(parents=True, exist_ok=True)

    def save_bm25_nodes(self, nodes: List[Any], db_name: str) -> None:
        bm25_path = self.paths["bm25"]
        bm25_path.mkdir(parents=True, exist_ok=True)
        file_path = bm25_path / "bm25_nodes.pkl"
        with open(file_path, "wb") as f:
            pickle.dump(nodes, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_bm25_nodes(self, db_name: str) -> Optional[List[Any]]:
        file_path = self.paths["bm25"] / "bm25_nodes.pkl"
        if not file_path.exists():
            return None
        with open(file_path, "rb") as f:
            return pickle.load(f)
