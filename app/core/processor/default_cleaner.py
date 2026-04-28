import re
from typing import List, Optional

from .base_cleaner import BaseTextCleaner


class DefaultCleaner(BaseTextCleaner):
    """Базовая очистка для TXT и неизвестных форматов."""

    @property
    def supported_extensions(self) -> List[str]:
        return ["*"]  # ловит всё, что не попало в другие очистители

    def clean(self, text: str, metadata: Optional[dict] = None) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
