import os
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.retrievers import QueryFusionRetriever

from .prompt_config import PromptConfig
from .storage_manager import StorageManager
from .index_registry import IndexRegistry
from .chat_session_manager import ChatSessionManager
from .llama_settings import configure_llama_settings
from .reranker import Reranker


class RAGService:
    def __init__(self, db_name: str):
        configure_llama_settings()
        self.prompts = PromptConfig.load()
        self.storage = StorageManager(db_name)
        self.indices = IndexRegistry(self.storage, self.prompts)
        self.reranker = Reranker()
        self.chat_mgr = ChatSessionManager(self.indices, self.prompts, self.reranker)

    # ─── Индексация ───
    def add_index(self, type_index: str, path: str, overwrite: bool = False):
        if type_index == "tree":
            self.indices.add("tree", path, overwrite=overwrite)
        elif type_index == "vector":
            self.indices.add("vector", path, overwrite=overwrite)
        elif type_index == "kg":
            self.indices.add("kg", path, overwrite=overwrite)
        elif type_index == "bm25":
            self.indices.add("bm25", path, overwrite=overwrite)
        else:
            raise ValueError(
                f"Неизвестный type_index: {type_index}. Доступны для индексации: tree, vector, kg, bm25"
            )

    def add_tree_index(self, path: str, overwrite: bool = False):
        self.add_index("tree", path, overwrite=overwrite)

    def add_vector_index(self, path: str, overwrite: bool = False):
        self.add_index("vector", path, overwrite=overwrite)

    def add_kg_index(self, path: str, overwrite: bool = False):
        self.add_index("kg", path, overwrite=overwrite)

    def add_bm25_index(self, path: str, overwrite: bool = False):
        self.add_index("bm25", path, overwrite=overwrite)

    # ─── Stateless запросы ───
    def query(
        self,
        text: str,
        type_index: str = "vector",
        top_k: int = 10,
        temp: float = 0.1,
        mt: int = 192,
        mode: str = "hybrid",
    ) -> str:
        if type_index == "tree":
            return self.query_tree(text=text, top_k=top_k, temp=temp, mt=mt)
        elif type_index == "vector":
            return self.query_vector(text=text, top_k=top_k, temp=temp, mt=mt)
        elif type_index == "kg":
            return self.query_kg(text=text, top_k=top_k, temp=temp, mt=mt, mode=mode)
        elif type_index == "bm25":
            return self.query_bm25(text=text, top_k=top_k)
        elif type_index == "hybrid":
            return self.query_hybrid(text=text, top_k=top_k)
        else:
            raise ValueError(
                f"Неизвестный type_index: {type_index}. Доступны: tree, vector, kg, bm25, hybrid"
            )

    def query_tree(self, text: str, top_k=10, temp=0.1, mt=192) -> str:
        idx = self.indices.get("tree")
        return idx.as_query_engine(
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.prompts["custom_template"],
            refine_template=self.prompts["refine_template"],
            query_template=self.prompts["tree_query_template"],
            query_template_multiple=self.prompts["tree_query_template_multiple"],
            node_postprocessors=self.reranker.get_postprocessors(),
        ).query(text)

    def query_vector(self, text: str, top_k=10, temp=0.1, mt=192) -> str:
        idx = self.indices.get("vector")
        return idx.as_query_engine(
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.prompts["custom_template"],
            node_postprocessors=self.reranker.get_postprocessors(),
        ).query(text)

    def query_kg(self, text: str, top_k=10, temp=0.1, mt=192, mode="hybrid") -> str:
        idx = self.indices.get("kg")
        return idx.as_query_engine(
            retriever_mode=mode,
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.prompts["custom_template"],
            refine_template=self.prompts["refine_template"],
            query_keyword_extract_template=self.prompts["kg_query_keyword_template"],
            node_postprocessors=self.reranker.get_postprocessors(),
        ).query(text)

    def query_bm25(self, text: str, top_k=10) -> str:
        retriever = self.indices.get("bm25")
        if hasattr(retriever, "similarity_top_k"):
            retriever.similarity_top_k = top_k
        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=self.prompts["custom_template"],
            node_postprocessors=self.reranker.get_postprocessors(),
        ).query(text)

    def query_hybrid(self, text: str, top_k=10) -> str:
        vector_index = self.indices.get("vector")
        vector_retriever = vector_index.as_retriever(similarity_top_k=top_k)

        bm25_retriever = self.indices.get("bm25")
        if hasattr(bm25_retriever, "similarity_top_k"):
            bm25_retriever.similarity_top_k = top_k

        retriever = QueryFusionRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            retriever_weights=[0.6, 0.4],
            similarity_top_k=top_k,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        return RetrieverQueryEngine.from_args(
            retriever=retriever,
            text_qa_template=self.prompts["custom_template"],
            node_postprocessors=self.reranker.get_postprocessors(),
        ).query(text)

    # ─── Stateful чат ───
    def get_chat_engine(
        self, session_id: str, index_type="vector", top_k=10, token_limit=4000
    ):
        return self.chat_mgr.get_engine(session_id, index_type, top_k, token_limit)

    def chat(
        self, text: str, session_id: str | None, index_type="vector", top_k=3
    ) -> str:
        if session_id is None or not session_id.strip():
            return self.query(text=text, type_index=index_type, top_k=top_k)

        engine = self.get_chat_engine(session_id, index_type, top_k)
        resp = engine.chat(text)
        chat_store_path = os.path.join(
            self.storage.paths["memory"], session_id, "chat_store.json"
        )
        if hasattr(engine._memory.chat_store, "persist"):
            engine._memory.chat_store.persist(persist_path=chat_store_path)
        return resp.response
