import re
from typing import List, Optional

from .base_cleaner import BaseTextCleaner


class MarkdownCleaner(BaseTextCleaner):
    """Очистка Markdown: убирает синтаксис, оставляя семантику."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".md", ".markdown"]

    def clean(self, text: str, metadata: Optional[dict] = None) -> str:
        text = re.sub(r"!\[.*?\]\(.*?\)", "", text)  # изображения
        text = re.sub(r"\[(.*?)\]\(.*?\)", r"\1", text)  # ссылки → текст
        text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)  # заголовки
        text = re.sub(r"[*_`~]{1,3}", "", text)  # жирный/курсив/код
        text = re.sub(r"[-*+]\s+", "• ", text, flags=re.MULTILINE)  # маркеры списка
        return re.sub(r"\n{3,}", "\n\n", text).strip()
