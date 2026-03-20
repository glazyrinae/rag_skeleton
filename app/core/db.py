import os
import logging
import shutil
import time
from typing import List, Dict, Any, Optional, Iterator, Tuple
from deeplake import VectorStore
from fastembed import TextEmbedding
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import Document as LlamaDocument


logger = logging.getLogger(__name__)

EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
EMBEDDING_CACHE_DIR = "/app/embedding_models"
DEFAULT_CHUNK_SIZE = 1000
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_EMBEDDING_BATCH_SIZE = 64
SUPPORTED_EXTENSIONS = {".py", ".md", ".txt", ".pdf", ".html"}
IGNORED_DIRS = {".", "__pycache__", "venv", "env", "node_modules", ".git"}


class VectorDB:
    """Векторная база данных на DeepLake с LlamaIndex"""

    def __init__(
        self,
        path: str = "/app/deeplake_data",
        dataset_name: str = "default",
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
    ):
        self.base_path = path
        self.default_dataset = dataset_name
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_model = None
        self.vector_stores: Dict[str, VectorStore] = {}

        # Инициализируем сплиттер из LlamaIndex
        self.node_parser = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=" ",
            paragraph_separator="\n\n",
            secondary_chunking_regex="[^,.;。]+[,.;。]?",
        )

    def _init_embedding_model(self) -> TextEmbedding:
        """Инициализация легкой модели эмбеддингов через FastEmbed"""
        if self.embedding_model is None:
            os.makedirs(EMBEDDING_CACHE_DIR, exist_ok=True)
            logger.info(
                "Инициализация embedding-модели model=%s cache_dir=%s",
                EMBEDDING_MODEL,
                EMBEDDING_CACHE_DIR,
            )
            try:
                self.embedding_model = TextEmbedding(
                    model_name=EMBEDDING_MODEL,
                    cache_dir=EMBEDDING_CACHE_DIR,
                    # onnx_execution_provider="CPUExecutionProvider"  # по умолчанию
                )
            except Exception as e:
                if self._is_broken_embedding_cache_error(e):
                    logger.warning(
                        "Обнаружен поврежденный кэш эмбеддинг-модели %s, очищаем и повторяем загрузку",
                        EMBEDDING_MODEL,
                    )
                    self._clear_embedding_model_cache(EMBEDDING_MODEL)
                    self.embedding_model = TextEmbedding(
                        model_name=EMBEDDING_MODEL,
                        cache_dir=EMBEDDING_CACHE_DIR,
                    )
                else:
                    raise
            logger.info("Embedding-модель готова model=%s", EMBEDDING_MODEL)

        return self.embedding_model

    def _is_broken_embedding_cache_error(self, error: Exception) -> bool:
        """Определение ошибки битого кэша ONNX-модели."""
        error_text = str(error)
        return "NO_SUCHFILE" in error_text or "File doesn't exist" in error_text

    def _clear_embedding_model_cache(self, model_name: str) -> None:
        """Удаление битого кэша модели FastEmbed перед повторной загрузкой."""
        model_slug = model_name.split("/")[-1]
        cache_entries = [
            os.path.join(EMBEDDING_CACHE_DIR, entry)
            for entry in os.listdir(EMBEDDING_CACHE_DIR)
            if model_slug in entry
        ]

        for cache_entry in cache_entries:
            if os.path.isdir(cache_entry):
                shutil.rmtree(cache_entry, ignore_errors=True)
            elif os.path.exists(cache_entry):
                os.remove(cache_entry)

    def _get_embeddings(
        self,
        texts: List[str],
        input_type: str = "passage",
        batch_size: int = DEFAULT_EMBEDDING_BATCH_SIZE,
    ) -> List[List[float]]:
        """Получение эмбеддингов для текстов"""
        started_at = time.perf_counter()
        model = self._init_embedding_model()
        logger.info(
            "Старт генерации эмбеддингов model=%s input_type=%s texts=%s",
            EMBEDDING_MODEL,
            input_type,
            len(texts),
        )
        if not texts:
            return []

        embed_fn = model.embed
        if input_type == "query" and hasattr(model, "query_embed"):
            embed_fn = model.query_embed
        elif input_type == "passage" and hasattr(model, "passage_embed"):
            embed_fn = model.passage_embed

        vectors: List[List[float]] = []
        total_texts = len(texts)

        total_batches = (total_texts + batch_size - 1) // batch_size

        for start_idx in range(0, total_texts, batch_size):
            batch_number = start_idx // batch_size + 1
            batch_started_at = time.perf_counter()
            batch_texts = texts[start_idx : start_idx + batch_size]

            logger.info(
                "Embedding batch started batch=%s/%s size=%s processed=%s/%s",
                batch_number,
                total_batches,
                len(batch_texts),
                start_idx,
                total_texts,
            )

            batch_embeddings = list(embed_fn(batch_texts))
            batch_vectors = [embedding.tolist() for embedding in batch_embeddings]
            vectors.extend(batch_vectors)

            logger.info(
                "Embedding batch finished batch=%s/%s size=%s elapsed=%.2fs processed=%s/%s",
                batch_number,
                total_batches,
                len(batch_vectors),
                time.perf_counter() - batch_started_at,
                min(start_idx + len(batch_texts), total_texts),
                total_texts,
            )

        logger.info(
            "Эмбеддинги готовы count=%s dim=%s elapsed=%.2fs",
            len(vectors),
            len(vectors[0]) if vectors else 0,
            time.perf_counter() - started_at,
        )
        return vectors

    def _build_exclude_patterns(self) -> List[str]:
        """Паттерны для исключения служебных директорий при загрузке."""
        patterns = []

        for ignored_dir in IGNORED_DIRS:
            if ignored_dir == ".":
                continue
            patterns.append(f"**/{ignored_dir}")
            patterns.append(f"**/{ignored_dir}/**")

        return patterns

    def _get_dataset_path(self, dataset_name: str) -> str:
        """Формирование полного пути к датасету."""
        return f"{self.base_path}/{dataset_name}"

    def _create_vector_store(
        self,
        dataset_name: str,
        overwrite: bool = False,
        read_only: bool = False,
    ) -> VectorStore:
        """Создание подключения к DeepLake dataset."""
        dataset_path = self._get_dataset_path(dataset_name)
        logger.info(f"📁 Открываем датасет: {dataset_path}")
        return VectorStore(
            path=dataset_path,
            overwrite=overwrite,
            read_only=read_only,
            verbose=False,
        )

    def _get_vector_store(
        self,
        dataset_name: Optional[str] = None,
        overwrite: bool = False,
        read_only: bool = False,
    ) -> VectorStore:
        """Получение кэшированного VectorStore для нужного датасета."""
        dataset_name = dataset_name or self.default_dataset

        if overwrite or read_only:
            return self._create_vector_store(
                dataset_name=dataset_name,
                overwrite=overwrite,
                read_only=read_only,
            )

        if dataset_name not in self.vector_stores:
            self.vector_stores[dataset_name] = self._create_vector_store(dataset_name)

        return self.vector_stores[dataset_name]

    def _iter_documents(self, project_path: str) -> Iterator[Dict[str, Any]]:
        """Потоковая выдача документов через штатные readers LlamaIndex."""
        if not os.path.isdir(project_path):
            raise ValueError(f"Каталог не найден: {project_path}")

        logger.info(
            "Старт сканирования project_path=%s extensions=%s",
            project_path,
            ",".join(sorted(SUPPORTED_EXTENSIONS)),
        )
        reader = SimpleDirectoryReader(
            input_dir=project_path,
            recursive=True,
            required_exts=sorted(SUPPORTED_EXTENSIONS),
            exclude=self._build_exclude_patterns(),
            filename_as_id=True,
        )
        if hasattr(reader, "iter_data"):
            document_batches = reader.iter_data(show_progress=True)
        else:
            document_batches = [reader.load_data(show_progress=True)]

        for llama_documents in document_batches:
            for doc in llama_documents:
                source = doc.metadata.get("file_path") or doc.metadata.get("filename")
                file_ext = os.path.splitext(source or "")[1].lower()
                yield {
                    "text": doc.text,
                    "metadata": {
                        "source": source,
                        "file_type": file_ext,
                        **doc.metadata,
                    },
                }

    def _iter_document_chunks(
        self, document: Dict[str, Any]
    ) -> Iterator[Tuple[str, Dict[str, Any]]]:
        """Потоковая выдача чанков одного документа."""
        llama_doc = LlamaDocument(text=document["text"], metadata=document["metadata"])
        nodes = self.node_parser.get_nodes_from_documents([llama_doc])

        for node in nodes:
            metadata = document["metadata"].copy()
            metadata.update(
                {
                    "node_id": node.node_id,
                    "chunk_start": (
                        node.start_char_idx if hasattr(node, "start_char_idx") else None
                    ),
                    "chunk_end": (
                        node.end_char_idx if hasattr(node, "end_char_idx") else None
                    ),
                }
            )
            yield node.text, metadata

    def _flush_chunks_batch(
        self,
        vector_store: VectorStore,
        dataset_name: str,
        batch_chunks: List[str],
        batch_metadatas: List[Dict[str, Any]],
        batch_number: int,
        total_chunks_written: int,
    ) -> int:
        """Генерация эмбеддингов и запись одного батча чанков."""
        if not batch_chunks:
            return total_chunks_written

        batch_started_at = time.perf_counter()
        logger.info(
            "Запись батча batch=%s size=%s already_written=%s",
            batch_number,
            len(batch_chunks),
            total_chunks_written,
        )

        embeddings = self._get_embeddings(batch_chunks)

        if not (len(batch_chunks) == len(batch_metadatas) == len(embeddings)):
            raise ValueError(
                "Размеры батча не совпадают: "
                f"{len(batch_chunks)=}, {len(batch_metadatas)=}, {len(embeddings)=}"
            )

        try:
            vector_store.add(
                text=batch_chunks,
                embedding=embeddings,
                metadata=batch_metadatas,
            )
        except Exception:
            logger.exception(
                "Ошибка при записи батча batch=%s dataset=%s",
                batch_number,
                dataset_name,
            )
            raise

        total_chunks_written += len(batch_chunks)
        logger.info(
            "Батч записан batch=%s size=%s total_written=%s elapsed=%.2fs",
            batch_number,
            len(batch_chunks),
            total_chunks_written,
            time.perf_counter() - batch_started_at,
        )
        return total_chunks_written

    def add_docs_to_db(
        self,
        dataset_name: str,
        project_path: str,
        batch_size: int = 100,
        overwrite: bool = True,
    ):
        """Добавление документов в DeepLake с использованием LlamaIndex для сплиттинга"""
        started_at = time.perf_counter()
        logger.info(
            "Старт индексации dataset=%s project_path=%s batch_size=%s overwrite=%s",
            dataset_name,
            project_path,
            batch_size,
            overwrite,
        )

        dataset_path = self._get_dataset_path(dataset_name)
        vector_store = self._get_vector_store(
            dataset_name=dataset_name,
            overwrite=overwrite,
        )
        self.vector_stores[dataset_name] = vector_store

        batch_number = 0
        documents_count = 0
        total_chunks = 0
        total_chunks_written = 0
        pending_chunks: List[str] = []
        pending_metadatas: List[Dict[str, Any]] = []

        for document in self._iter_documents(project_path):
            documents_count += 1
            document_started_at = time.perf_counter()
            source = document["metadata"].get("source", "<unknown>")
            document_chunks = 0

            logger.info(
                "Обработка документа document=%s source=%s",
                documents_count,
                source,
            )

            for chunk_text, chunk_metadata in self._iter_document_chunks(document):
                pending_chunks.append(chunk_text)
                pending_metadatas.append(chunk_metadata)
                document_chunks += 1
                total_chunks += 1

                if len(pending_chunks) < batch_size:
                    continue

                batch_number += 1
                total_chunks_written = self._flush_chunks_batch(
                    vector_store=vector_store,
                    dataset_name=dataset_name,
                    batch_chunks=pending_chunks,
                    batch_metadatas=pending_metadatas,
                    batch_number=batch_number,
                    total_chunks_written=total_chunks_written,
                )
                pending_chunks = []
                pending_metadatas = []

            logger.info(
                "Документ разбит document=%s source=%s chunks=%s elapsed=%.2fs pending_chunks=%s total_chunks=%s",
                documents_count,
                source,
                document_chunks,
                time.perf_counter() - document_started_at,
                len(pending_chunks),
                total_chunks,
            )

        if pending_chunks:
            batch_number += 1
            total_chunks_written = self._flush_chunks_batch(
                vector_store=vector_store,
                dataset_name=dataset_name,
                batch_chunks=pending_chunks,
                batch_metadatas=pending_metadatas,
                batch_number=batch_number,
                total_chunks_written=total_chunks_written,
            )

        logger.info(
            "Прогресс индексации documents=%s chunks=%s written=%s",
            documents_count,
            total_chunks,
            total_chunks_written,
        )

        if documents_count == 0:
            raise ValueError("Нет документов для обработки")
        if total_chunks == 0:
            raise ValueError("Не удалось создать чанки из документов")

        logger.info(
            "Индексация завершена dataset=%s documents=%s chunks=%s elapsed=%.2fs",
            dataset_name,
            documents_count,
            total_chunks,
            time.perf_counter() - started_at,
        )

        # Возвращаем информацию о датасете
        return {
            "dataset_path": dataset_path,
            "chunks_count": total_chunks,
            "documents_count": documents_count,
        }

    def search(
        self,
        query: str,
        dataset_name: Optional[str] = None,
        k: int = 5,
        return_scores: bool = True,
    ) -> List[Dict[str, Any]]:
        """Поиск в DeepLake"""
        started_at = time.perf_counter()
        if dataset_name is None:
            dataset_name = self.default_dataset

        logger.info("Старт поиска dataset=%s k=%s", dataset_name, k)
        vector_store = self._get_vector_store(
            dataset_name=dataset_name,
            read_only=True,
        )

        # Получаем эмбеддинг запроса
        query_embedding = self._get_embeddings([query], input_type="query")[0]

        # Поиск
        results = vector_store.search(
            embedding=query_embedding, k=k, return_tensors=True
        )

        # Форматируем результаты
        formatted_results = []
        for i in range(len(results["text"])):
            result = {
                "text": results["text"][i],
                "metadata": results["metadata"][i] if "metadata" in results else {},
                "score": (
                    results["score"][i]
                    if "score" in results and return_scores
                    else None
                ),
            }
            formatted_results.append(result)

        logger.info(
            "Поиск завершен dataset=%s results=%s elapsed=%.2fs",
            dataset_name,
            len(formatted_results),
            time.perf_counter() - started_at,
        )
        return formatted_results
