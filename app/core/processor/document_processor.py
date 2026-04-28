from pathlib import Path
from typing import List, Optional

from llama_index.core.node_parser import SentenceSplitter
from natasha import MorphVocab, Doc, Segmenter, NewsEmbedding, NewsMorphTagger

from .base_cleaner import BaseTextCleaner
from .default_cleaner import DefaultCleaner
from .html_cleaner import HtmlCleaner
from .md_cleaner import MarkdownCleaner
from .pdf_cleaner import PdfCleaner


class DocumentProcessor:
    """
    Центральный процессор документов.
    Автоматически выбирает очиститель по расширению файла из метаданных.
    """

    def __init__(self, cleaners: Optional[List[BaseTextCleaner]] = None):
        self._cleaners: dict[str, BaseTextCleaner] = {}

        # Регистрируем очистители по умолчанию
        self._register_defaults()

        # Если переданы кастомные, добавляем их (перезапишут дефолтные при совпадении расширений)
        if cleaners:
            for cleaner in cleaners:
                self.register(cleaner)

        # Инициализация NLP (однократно)
        self._morph_vocab = MorphVocab()
        self._segmenter = Segmenter()
        self._morph_emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(self._morph_emb)

    def _register_defaults(self):
        self.register(PdfCleaner())
        self.register(MarkdownCleaner())
        self.register(HtmlCleaner())
        self.register(DefaultCleaner())

    def register(self, cleaner: BaseTextCleaner) -> None:
        """Добавляет новый очиститель в реестр."""
        for ext in cleaner.supported_extensions:
            self._cleaners[ext] = cleaner

    def _get_cleaner(self, file_path: Optional[str] = None) -> BaseTextCleaner:
        """Возвращает очиститель по расширению файла или дефолтный."""
        if not file_path:
            return self._cleaners.get("*", DefaultCleaner())

        ext = Path(file_path).suffix.lower()
        return self._cleaners.get(ext, self._cleaners.get("*", DefaultCleaner()))

    def clean_text(self, text: str, file_path: Optional[str] = None) -> str:
        """Очищает текст, автоматически выбирая стратегию."""
        cleaner = self._get_cleaner(file_path)
        return cleaner.clean(text)

    def process_batch(self, batch: list) -> list:
        """Применяет очистку к батчу документов перед индексацией."""
        for doc in batch:
            if hasattr(doc, "text") and doc.text:
                file_path = getattr(doc, "metadata", {}).get("file_path", None)
                doc.text = self.clean_text(doc.text, file_path)
        return batch

    def get_parser(
        self, chunk_size: int = 384, chunk_overlap: int = 48
    ) -> SentenceSplitter:
        """Возвращает настроенный сплиттер для разбиения на чанки."""
        return SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            paragraph_separator="\n\n",
        )

    def lemmatize(self, text: str) -> List[str]:
        """Лемматизация русского текста для BM25/поиска."""
        doc = Doc(text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)

        lemmas: List[str] = []
        for token in doc.tokens:
            token.lemmatize(self._morph_vocab)
            lemma = token.lemma.lower()
            if lemma and not lemma.isspace():
                lemmas.append(lemma)
        return lemmas
