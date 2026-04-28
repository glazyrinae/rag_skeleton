"""
Реестр индексов. Управляет жизненным циклом индексов,
делегируя хранение и обработку текста внешним компонентам.
"""

import os
from typing import Optional, Any
from llama_index.core import SimpleDirectoryReader, StorageContext, Settings
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.tree import TreeIndex
from llama_index.core import VectorStoreIndex
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.retrievers.bm25 import BM25Retriever

from core.storage.storage_manager import StorageManager
from core.processor.document_processor import DocumentProcessor


class IndexRegistry:
    def __init__(
        self, storage: StorageManager, doc_processor: DocumentProcessor, prompts: dict
    ):
        self.storage = storage
        self.prompts = prompts
        self.indices: dict[str, Optional[Any]] = {
            "vector": None,
            "tree": None,
            "kg": None,
            "bm25": None,
        }
        # Вся работа с текстом (очистка, парсинг, лемматизация) вынесена сюда
        self.doc_processor = doc_processor

    def _reset_index(self, index_type: str) -> None:
        if index_type not in self.indices:
            raise ValueError(
                f"Неизвестный index_type: {index_type}. Доступны: tree, vector, kg, bm25"
            )
        self.indices[index_type] = None
        # Удаление делегируется бэкенду
        self.storage.reset_index(index_type)

    def add(self, index_type: str, path_to_docs: str, overwrite: bool = False) -> None:
        if overwrite:
            self._reset_index(index_type)

        docs = SimpleDirectoryReader(path_to_docs, recursive=True)
        parser = self.doc_processor.get_parser()

        if index_type == "bm25":
            self._build_bm25(docs, parser)
            return

        ctx = self.storage.build_context(index_type)

        if index_type == "tree":
            self._build_tree(ctx, docs, parser)
        elif index_type == "vector":
            self._build_vector(ctx, docs, parser)
        elif index_type == "kg":
            self._build_kg(ctx, docs, parser)
        else:
            raise ValueError(
                f"Неизвестный index_type: {index_type}. Доступны: tree, vector, kg, bm25"
            )

        self.storage.persist_context(ctx, index_type)

    def _iter_docs(self, docs: SimpleDirectoryReader):
        """Ленивая итерация по батчам с автоматической очисткой под формат файла."""
        for batch in docs.iter_data():
            yield self.doc_processor.process_batch(batch)

    def _build_tree(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser
    ) -> None:
        docs_iter = self._iter_docs(docs)

        if self.storage.index_exists("tree"):
            self.indices["tree"] = load_index_from_storage(ctx, index_id="tree_index")
            for batch in docs_iter:
                for doc in batch:
                    self.indices["tree"].insert(doc)
        else:
            if first_batch := next(docs_iter, None):
                self.indices["tree"] = TreeIndex.from_documents(
                    first_batch,
                    storage_context=ctx,
                    transformations=[parser],
                    summary_template=self.prompts["summary_prompt"],
                )
                self.indices["tree"].set_index_id("tree_index")
                for batch in docs_iter:
                    for doc in batch:
                        self.indices["tree"].insert(doc)

    def _build_vector(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser
    ) -> None:
        docs_iter = self._iter_docs(docs)

        if self.storage.index_exists("vector"):
            self.indices["vector"] = load_index_from_storage(
                ctx, index_id="vector_index"
            )
            for batch in docs_iter:
                for doc in batch:
                    self.indices["vector"].insert(doc)
        else:
            if first_batch := next(docs_iter, None):
                self.indices["vector"] = VectorStoreIndex.from_documents(
                    first_batch, storage_context=ctx, transformations=[parser]
                )
                self.indices["vector"].set_index_id("vector_index")
                for batch in docs_iter:
                    for doc in batch:
                        self.indices["vector"].insert(doc)

    def _build_kg(
        self, ctx: StorageContext, docs: SimpleDirectoryReader, parser
    ) -> None:
        docs_iter = self._iter_docs(docs)

        if self.storage.index_exists("kg"):
            self.indices["kg"] = load_index_from_storage(ctx, index_id="kg_index")
            for batch in docs_iter:
                for doc in batch:
                    self.indices["kg"].insert(doc)
        else:
            if first_batch := next(docs_iter, None):
                self.indices["kg"] = KnowledgeGraphIndex.from_documents(
                    first_batch,
                    storage_context=ctx,
                    transformations=[parser],
                    llm=Settings.llm,
                    embed_model=Settings.embed_model,
                    kg_triplet_extract_template=self.prompts[
                        "kg_triplet_extract_template"
                    ],
                    max_triplets_per_chunk=int(
                        os.getenv("KG_MAX_TRIPLETS_PER_CHUNK", "8")
                    ),
                    include_embeddings=True,
                )
                self.indices["kg"].set_index_id("kg_index")
                for batch in docs_iter:
                    for doc in batch:
                        self.indices["kg"].insert(doc)

    def _build_bm25(self, docs: SimpleDirectoryReader, parser) -> None:
        nodes = []
        for batch in self._iter_docs(docs):
            nodes.extend(parser.get_nodes_from_documents(batch))

        if not nodes:
            raise RuntimeError("Не найдено документов для построения BM25 индекса.")

        self.indices["bm25"] = BM25Retriever.from_defaults(
            nodes=nodes,
            similarity_top_k=10,
            tokenizer=self.doc_processor.lemmatize,
        )
        # BM25 сохраняется отдельно, т.к. не использует StorageContext
        self.storage.save_bm25_nodes(nodes)

    def _load_persisted(self, index_type: str) -> Optional[Any]:
        if not self.storage.index_exists(index_type):
            return None

        ctx = self.storage.build_context(index_type)
        return load_index_from_storage(ctx, index_id=f"{index_type}_index")

    def get(self, index_type: str) -> Any:
        if self.indices[index_type] is None:
            loaded = self._load_persisted(index_type)
            if loaded is not None:
                self.indices[index_type] = loaded
            elif index_type == "bm25":
                # BM25 загружается напрямую, минуя StorageContext
                nodes = self.storage.load_bm25_nodes()
                if nodes:
                    self.indices["bm25"] = BM25Retriever.from_defaults(
                        nodes=nodes,
                        similarity_top_k=10,
                        tokenizer=self.doc_processor.lemmatize,
                    )
                else:
                    raise RuntimeError(f"Индекс '{index_type}' не найден в хранилище.")
            else:
                raise RuntimeError(
                    f"Индекс '{index_type}' не создан. Вызовите add() перед использованием."
                )
        return self.indices[index_type]
