import re
from typing import List, Optional

from .base_cleaner import BaseTextCleaner


class HtmlCleaner(BaseTextCleaner):
    """Очистка HTML: удаление тегов, нормализация пробелов."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".html", ".htm"]

    def clean(self, text: str, metadata: Optional[dict] = None) -> str:
        # Простая замена тегов (для продакшена лучше использовать beautifulsoup4)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"&\w+;", " ", text)  # HTML-сущности
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
