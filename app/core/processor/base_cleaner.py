"""
Обработка и очистка текстов из различных форматов.
Архитектура: Strategy + Registry. Легко добавлять новые форматы.
"""

from abc import ABC, abstractmethod
from typing import List, Optional


class BaseTextCleaner(ABC):
    """Интерфейс для формат-специфичной очистки текста."""

    @property
    @abstractmethod
    def supported_extensions(self) -> List[str]:
        """Список расширений, которые обрабатывает этот очиститель (например, ['.pdf'])."""
        pass

    @abstractmethod
    def clean(self, text: str, metadata: Optional[dict] = None) -> str:
        """Возвращает очищенный текст."""
        pass
