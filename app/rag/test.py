import os

from llama_index.core import Settings, SimpleDirectoryReader, StorageContext
from llama_index.core.chat_engine import CondensePlusContextChatEngine
from llama_index.core.indices import load_index_from_storage
from llama_index.core.indices.knowledge_graph import KnowledgeGraphIndex
from llama_index.core.indices.tree import TreeIndex
from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.prompts import PromptTemplate
from llama_index.core.storage.chat_store import SimpleChatStore
from llama_index.core.storage.docstore import SimpleDocumentStore
from llama_index.core.storage.index_store import SimpleIndexStore
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.deeplake import DeepLakeVectorStore
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama

# from rag.test import VectorDB

# db = VectorDB("gost")
# db.add_vector_index("/app/data/gost")
# print(db.query_vector("что говорится в пункте 4.11"))
# print(db.chat("что говорится в пункте 4.11", session_id="user_1", index_type="kg"))
# print(db.chat("уточни коротко", session_id="user_1", index_type="kg"))

Settings.embed_model = OllamaEmbedding(
    model_name=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11438"),
    embed_batch_size=int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "32")),
    ollama_additional_kwargs={
        "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "8")),
    },
)
Settings.llm = Ollama(
    model=os.getenv("OLLAMA_MODEL", "llama3"),
    base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11438"),
    temperature=0.1,
    request_timeout=float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")),
    context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "16384")),
    additional_kwargs={
        "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "8")),
    },
)


class VectorDB:

    def __init__(self, db_name: str):
        self.index = None  # backward compatibility (tree index)
        self.tree_index = None
        self.vector_index = None
        self.kg_index = None
        self.db_name = db_name
        self.path_to_db = f"/app/deeplake_data/{db_name}"
        self.vector_store = DeepLakeVectorStore(
            dataset_path=self.path_to_db, overwrite=False, read_only=False
        )
        self.docstore_path = f"/app/deeplake_data/{db_name}_docstore"
        self.tree_indexstore_path = f"/app/deeplake_data/{db_name}_tree_indexstore"
        self.vector_indexstore_path = f"/app/deeplake_data/{db_name}_vector_indexstore"
        self.kg_storage_path = f"/app/deeplake_data/{db_name}_kg_storage"
        self.custom_template_str = """
                Ты — помощник и консультант. Отвечай строго на основе предоставленного контекста.
                Если в контексте нет ответа, так и скажи: "В документах нет информации по этому вопросу".
                Пиши сразу с сути, без вступлений и служебных формулировок.
                Запрещено использовать фразы: "Ответ", "В соответствии с запросом", "Новый ответ", "Я должен предоставить информацию".
                Не используй markdown-заголовки и оформление (**...**).
                Контекст:
                {context_str}

                Вопрос: {query_str}
                Ответ:
            """
        self.chat_context_prompt_str = """
                Ты — помощник и консультант. Отвечай строго на основе предоставленного контекста.
                Если в контексте нет ответа, так и скажи: "В документах нет информации по этому вопросу".
                Пиши сразу с сути, без вступлений и служебных формулировок.
                Запрещено использовать фразы: "Ответ", "В соответствии с запросом", "Новый ответ", "Я должен предоставить информацию".
                Не используй markdown-заголовки и оформление (**...**).

                Контекст:
                {context_str}

                Инструкция: ответь на последний вопрос пользователя только по контексту выше.
            """
        self.custom_template = PromptTemplate(self.custom_template_str)
        self.condense_prompt_str = (
            "Суммаризируй историю диалога на русском, сохранив ключевые факты, "
            "номера статей/пунктов и суть вопроса. Будь краток.\n"
            "История:\n{chat_history}\nВопрос: {question}"
        )
        self.refine_template = PromptTemplate(
            """
                Ты уточняешь уже сформированный ответ по новому контексту.
                Отвечай только по контексту, без внешних знаний.
                Если новый контекст не добавляет фактов, верни исходный ответ без изменений.
                Пиши сразу по сути, без вступлений и служебных блоков.
                Не используй фразы "Original Answer", "Новый ответ", "В соответствии с запросом".

                Вопрос: {query_str}
                Текущий ответ: {existing_answer}
                Новый контекст:
                {context_msg}

                Обновленный ответ:
                """
        )

        self.summary_prompt = PromptTemplate(
            "Суммаризируй максимально точно только по контексту.\n\n{context_str}\n\nСводка:"
        )
        self.tree_query_template = PromptTemplate(
            """
                Ниже список вариантов (1..{num_chunks}), каждый вариант — это краткая сводка узла.
                ---------------------
                {context_list}
                ---------------------
                Используя только варианты выше и не используя внешние знания,
                выбери один самый релевантный вопросу: "{query_str}".
                Ответ верни в формате: ANSWER: <number>
                """
        )
        self.tree_query_template_multiple = PromptTemplate(
            """
                Ниже список вариантов (1..{num_chunks}), каждый вариант — это краткая сводка узла.
                ---------------------
                {context_list}
                ---------------------
                Используя только варианты выше и не используя внешние знания,
                выбери не более {branching_factor} самых релевантных вопросу "{query_str}".
                Упорядочи от более релевантного к менее релевантному.
                Ответ верни в формате: ANSWER: <numbers>
                """
        )
        self.kg_triplet_extract_template = PromptTemplate(
            """
                Извлеки до {max_knowledge_triplets} триплетов знаний из текста.
                Формат: (субъект, отношение, объект)
                Не используй внешние знания, только текст ниже.
                ---------------------
                {text}
                ---------------------
                Триплеты:
                """
        )
        self.kg_query_keyword_template = PromptTemplate(
            """
                Из вопроса извлеки до {max_keywords} ключевых слов для поиска по графу знаний.
                Избегай стоп-слов.
                ---------------------
                {question}
                ---------------------
                Формат ответа: KEYWORDS: <слово1, слово2, ...>
                """
        )

    def _build_storage_context(self, indexstore_path: str) -> StorageContext:
        docstore_file = os.path.join(self.docstore_path, "docstore.json")
        indexstore_file = os.path.join(indexstore_path, "index_store.json")
        return StorageContext.from_defaults(
            vector_store=self.vector_store,
            docstore=(
                SimpleDocumentStore.from_persist_dir(self.docstore_path)
                if os.path.exists(docstore_file)
                else SimpleDocumentStore()
            ),
            index_store=(
                SimpleIndexStore.from_persist_dir(indexstore_path)
                if os.path.exists(indexstore_file)
                else SimpleIndexStore()
            ),
        )

    def _persist_storage_context(
        self, storage_context: StorageContext, indexstore_path: str
    ) -> None:
        docstore_file = os.path.join(self.docstore_path, "docstore.json")
        indexstore_file = os.path.join(indexstore_path, "index_store.json")
        os.makedirs(self.docstore_path, exist_ok=True)
        os.makedirs(indexstore_path, exist_ok=True)
        storage_context.docstore.persist(persist_path=docstore_file)
        storage_context.index_store.persist(persist_path=indexstore_file)

    def _build_parser(self) -> SentenceSplitter:
        return SentenceSplitter(
            chunk_size=384, chunk_overlap=48, paragraph_separator="\n\n"
        )

    def add_tree_index(self, path_to_docs: str) -> None:
        documents = SimpleDirectoryReader(path_to_docs).load_data()
        storage_context = self._build_storage_context(self.tree_indexstore_path)
        parser = self._build_parser()
        indexstore_file = os.path.join(self.tree_indexstore_path, "index_store.json")

        if os.path.exists(indexstore_file):
            self.tree_index = load_index_from_storage(
                storage_context=storage_context,
                index_id="tree_index",
            )
            for doc in documents:
                self.tree_index.insert(doc)
        else:
            self.tree_index = TreeIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[parser],
                summary_template=self.summary_prompt,
            )
            self.tree_index.set_index_id("tree_index")

        self.index = self.tree_index
        self._persist_storage_context(storage_context, self.tree_indexstore_path)

    def add_vector_index(self, path_to_docs: str) -> None:
        documents = SimpleDirectoryReader(path_to_docs).load_data()
        storage_context = self._build_storage_context(self.vector_indexstore_path)
        parser = self._build_parser()
        indexstore_file = os.path.join(self.vector_indexstore_path, "index_store.json")

        if os.path.exists(indexstore_file):
            self.vector_index = load_index_from_storage(
                storage_context=storage_context,
                index_id="vector_index",
            )
            for doc in documents:
                self.vector_index.insert(doc)
        else:
            self.vector_index = VectorStoreIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[parser],
            )
            self.vector_index.set_index_id("vector_index")

        self._persist_storage_context(storage_context, self.vector_indexstore_path)

    def add_kg_index(self, path_to_docs: str) -> None:
        documents = SimpleDirectoryReader(path_to_docs).load_data()
        parser = self._build_parser()
        docstore_file = os.path.join(self.kg_storage_path, "docstore.json")
        indexstore_file = os.path.join(self.kg_storage_path, "index_store.json")
        graphstore_file = os.path.join(self.kg_storage_path, "graph_store.json")
        os.makedirs(self.kg_storage_path, exist_ok=True)
        has_persisted_kg = (
            os.path.exists(docstore_file)
            and os.path.exists(indexstore_file)
            and os.path.exists(graphstore_file)
        )
        storage_context = (
            StorageContext.from_defaults(persist_dir=self.kg_storage_path)
            if has_persisted_kg
            else StorageContext.from_defaults()
        )

        if has_persisted_kg:
            self.kg_index = load_index_from_storage(
                storage_context=storage_context,
                index_id="kg_index",
            )
            for doc in documents:
                self.kg_index.insert(doc)
        else:
            self.kg_index = KnowledgeGraphIndex.from_documents(
                documents,
                storage_context=storage_context,
                transformations=[parser],
                llm=Settings.llm,
                embed_model=Settings.embed_model,
                kg_triplet_extract_template=self.kg_triplet_extract_template,
                max_triplets_per_chunk=int(os.getenv("KG_MAX_TRIPLETS_PER_CHUNK", "8")),
                include_embeddings=True,
            )
            self.kg_index.set_index_id("kg_index")

        self.kg_index.storage_context.persist(persist_dir=self.kg_storage_path)

    def _ensure_tree_index_loaded(self) -> None:
        if self.tree_index is not None:
            self.index = self.tree_index
            return

        if not os.path.exists(self.path_to_db):
            raise RuntimeError(
                "Tree индекс не инициализирован. Сначала вызови add_tree_index(path_to_docs)."
            )

        storage_context = self._build_storage_context(self.tree_indexstore_path)
        self.tree_index = load_index_from_storage(
            storage_context=storage_context,
            index_id="tree_index",
        )

        self.index = self.tree_index

    def _ensure_vector_index_loaded(self) -> None:
        if self.vector_index is not None:
            return

        if not os.path.exists(self.path_to_db):
            raise RuntimeError(
                "Vector индекс не инициализирован. Сначала вызови add_vector_index(path_to_docs)."
            )

        storage_context = self._build_storage_context(self.vector_indexstore_path)
        self.vector_index = load_index_from_storage(
            storage_context=storage_context,
            index_id="vector_index",
        )

    def _ensure_kg_index_loaded(self) -> None:
        if self.kg_index is not None:
            return

        docstore_file = os.path.join(self.kg_storage_path, "docstore.json")
        indexstore_file = os.path.join(self.kg_storage_path, "index_store.json")
        graphstore_file = os.path.join(self.kg_storage_path, "graph_store.json")
        has_persisted_kg = (
            os.path.exists(docstore_file)
            and os.path.exists(indexstore_file)
            and os.path.exists(graphstore_file)
        )
        if not has_persisted_kg:
            raise RuntimeError(
                "KG индекс не инициализирован. Сначала вызови add_kg_index(path_to_docs)."
            )

        storage_context = StorageContext.from_defaults(persist_dir=self.kg_storage_path)
        self.kg_index = load_index_from_storage(
            storage_context=storage_context,
            index_id="kg_index",
        )

    def query_tree(
        self, text: str, top_k: int = 10, temp: float = 0, mt: int = 1024
    ) -> str:
        self._ensure_tree_index_loaded()
        query_engine = self.tree_index.as_query_engine(
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.custom_template,
            refine_template=self.refine_template,
            query_template=self.tree_query_template,
            query_template_multiple=self.tree_query_template_multiple,
        )
        return query_engine.query(text)

    def query_vector(
        self, text: str, top_k: int = 10, temp: float = 0, mt: int = 1024
    ) -> str:
        self._ensure_vector_index_loaded()
        query_engine = self.vector_index.as_query_engine(
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.custom_template,
        )
        return query_engine.query(text)

    def query_kg(
        self,
        text: str,
        top_k: int = 10,
        temp: float = 0,
        mt: int = 1024,
        retriever_mode: str = "hybrid",
    ) -> str:
        self._ensure_kg_index_loaded()
        query_engine = self.kg_index.as_query_engine(
            retriever_mode=retriever_mode,
            similarity_top_k=top_k,
            temperature=temp,
            num_output=mt,
            text_qa_template=self.custom_template,
            refine_template=self.refine_template,
            query_keyword_extract_template=self.kg_query_keyword_template,
        )
        return query_engine.query(text)

    def get_chat_engine(
        self,
        session_id: str,
        index_type: str = "vector",  # "vector" | "tree" | "kg"
        top_k: int = 10,
        token_limit: int = 4000,
    ) -> CondensePlusContextChatEngine:
        """Возвращает диалоговый движок с памятью для выбранного типа индекса."""

        if index_type == "vector":
            self._ensure_vector_index_loaded()
            retriever = self.vector_index.as_retriever(similarity_top_k=top_k)
        elif index_type == "tree":
            self._ensure_tree_index_loaded()
            retriever = self.tree_index.as_retriever(similarity_top_k=top_k)
        elif index_type == "kg":
            self._ensure_kg_index_loaded()
            retriever = self.kg_index.as_retriever(
                similarity_top_k=top_k, retriever_mode="hybrid"
            )
        else:
            raise ValueError(
                f"Неизвестный index_type: {index_type}. Доступны: vector, tree, kg"
            )

        chat_dir = f"/app/deeplake_data/memory/{session_id}"
        chat_store_path = os.path.join(chat_dir, "chat_store.json")
        os.makedirs(chat_dir, exist_ok=True)
        chat_store = SimpleChatStore.from_persist_path(chat_store_path)

        memory = ChatMemoryBuffer.from_defaults(
            token_limit=token_limit, chat_store=chat_store, chat_store_key="default"
        )

        return CondensePlusContextChatEngine.from_defaults(
            retriever=retriever,
            # memory=memory,
            llm=Settings.llm,
            context_prompt=self.chat_context_prompt_str,
            condense_prompt=self.condense_prompt_str,
            verbose=os.getenv("VERBOSE_CHAT", "false").lower() == "true",
        )

    def chat(
        self, text: str, session_id: str, index_type: str = "vector", top_k: int = 10
    ) -> str:
        """Однострочный вызов: вопрос + ответ с памятью."""
        engine = self.get_chat_engine(
            session_id=session_id, index_type=index_type, top_k=top_k
        )
        response = engine.chat(text)
        chat_store_path = os.path.join(
            f"/app/deeplake_data/memory/{session_id}", "chat_store.json"
        )
        if hasattr(engine._memory.chat_store, "persist"):
            engine._memory.chat_store.persist(persist_path=chat_store_path)
        return response.response
