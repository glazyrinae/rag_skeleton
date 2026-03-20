import logging
from typing import Any, Dict, List, Optional

from core.db import VectorDB
from core.llm import Llm


logger = logging.getLogger(__name__)

DEFAULT_TOP_K = 5
DEFAULT_MAX_CONTEXT_CHARS = 6000


class RAG:
    """Минимальный retrieval-augmented generation поверх VectorDB и Llm."""

    def __init__(
        self,
        db: VectorDB,
        llm: Llm,
        top_k: int = DEFAULT_TOP_K,
        max_context_chars: int = DEFAULT_MAX_CONTEXT_CHARS,
    ):
        self.db = db
        self.llm = llm
        self.top_k = top_k
        self.max_context_chars = max_context_chars

    def ask(
        self,
        question: str,
        schema: str = "core",
        top_k: Optional[int] = None,
        include_sources: bool = True,
    ) -> Dict[str, Any]:
        """Ищет релевантный контекст и отправляет его в LLM."""
        if not question.strip():
            raise ValueError("Question must not be empty")

        actual_top_k = top_k or self.top_k
        logger.info(
            "RAG request started schema=%s top_k=%s question_len=%s",
            schema,
            actual_top_k,
            len(question),
        )

        search_results = self.db.search(
            query=question,
            dataset_name=schema,
            k=actual_top_k,
            return_scores=True,
        )
        context_items = self._normalize_context(search_results)
        prompt = self._build_prompt(question=question, context_items=context_items)
        answer = self.llm.ask(prompt=prompt)

        logger.info(
            "RAG request finished schema=%s contexts=%s answer_len=%s",
            schema,
            len(context_items),
            len(answer),
        )

        response: Dict[str, Any] = {
            "answer": answer,
            "schema": schema,
            "question": question,
            "used_chunks": len(context_items),
        }
        if include_sources:
            response["sources"] = context_items

        return response

    def _normalize_context(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Обрезает контекст до разумного размера и приводит результаты к единому виду."""
        normalized: List[Dict[str, Any]] = []
        current_context_chars = 0

        for index, item in enumerate(results, start=1):
            text = (item.get("text") or "").strip()
            if not text:
                continue

            if current_context_chars >= self.max_context_chars:
                break

            remaining = self.max_context_chars - current_context_chars
            text = text[:remaining]
            metadata = item.get("metadata") or {}
            normalized.append(
                {
                    "index": index,
                    "text": text,
                    "score": item.get("score"),
                    "source": metadata.get("source") or metadata.get("file_name"),
                    "metadata": metadata,
                }
            )
            current_context_chars += len(text)

        return normalized

    def _build_prompt(
        self,
        question: str,
        context_items: List[Dict[str, Any]],
    ) -> str:
        """Собирает простой prompt для answer generation."""
        context_blocks = []
        for item in context_items:
            header = f"[Источник {item['index']}]"
            if item.get("source"):
                header += f" {item['source']}"
            context_blocks.append(f"{header}\n{item['text']}")

        context_text = "\n\n".join(context_blocks)
        if not context_text:
            context_text = "Контекст не найден."

        return (
            "Ты отвечаешь на вопрос по найденному контексту из базы знаний.\n"
            "Используй контекст как главный источник фактов.\n"
            "Если данных недостаточно, прямо так и скажи.\n"
            "Не выдумывай детали, которых нет в контексте.\n\n"
            f"Контекст:\n{context_text}\n\n"
            f"Вопрос: {question}\n"
            "Ответ:"
        )
