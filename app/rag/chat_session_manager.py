import os
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core import Settings

from .index_registry import IndexRegistry


class ChatSessionManager:
    def __init__(self, registry: IndexRegistry, prompts: dict, reranker=None):
        self.registry = registry
        self.prompts = prompts
        self.reranker = reranker

    def get_engine(
        self,
        session_id: str,
        index_type: str = "vector",
        top_k: int = 10,
        token_limit: int = 4000,
    ) -> CondensePlusContextChatEngine:
        index = self.registry.get(index_type)
        if index_type == "bm25":
            retriever = index
            if hasattr(retriever, "similarity_top_k"):
                retriever.similarity_top_k = top_k
        elif index_type == "kg":
            retriever = index.as_retriever(
                similarity_top_k=top_k,
                retriever_mode="hybrid",
            )
        else:
            retriever = index.as_retriever(similarity_top_k=top_k)

        chat_dir = os.path.join(self.registry.storage.paths["memory"], session_id)
        os.makedirs(chat_dir, exist_ok=True)
        chat_store = SimpleChatStore.from_persist_path(
            os.path.join(chat_dir, "chat_store.json")
        )

        memory = ChatMemoryBuffer.from_defaults(
            token_limit=token_limit, chat_store=chat_store, chat_store_key="default"
        )
        node_postprocessors = (
            self.reranker.get_postprocessors() if self.reranker is not None else []
        )

        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            memory=memory,
            llm=Settings.llm,
            context_prompt=self.prompts["chat_context_prompt_str"],
            condense_prompt=self.prompts["condense_prompt_str"],
            node_postprocessors=node_postprocessors,
            verbose=os.getenv("VERBOSE_CHAT", "false").lower() == "true",
        )
