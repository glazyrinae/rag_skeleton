import re
from typing import List, Optional

from .base_cleaner import BaseTextCleaner


class PdfCleaner(BaseTextCleaner):
    """Очистка артефактов парсинга PDF (переносы, мягкие дефисы, склейки)."""

    @property
    def supported_extensions(self) -> List[str]:
        return [".pdf"]

    def __init__(self):
        self._line_break_re = re.compile(r"(?<=[A-Za-zА-Яа-яЁё])\n(?=[A-Za-zА-Яа-яЁё])")
        self._glued_sentence_re = re.compile(r"([)].,;:!?])([A-ZА-ЯЁ])")
        self._lower_upper_glue_re = re.compile(r"(?<=[a-zа-яё])(?=[A-ZА-ЯЁ])")

    def clean(self, text: str, metadata: Optional[dict] = None) -> str:
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = text.replace("\u00ad", " ")  # мягкий перенос
        text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", " ", text)  # разрыв слова с дефисом
        text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
        text = self._line_break_re.sub(" ", text)  # склейка слов, разорванных \n
        text = self._glued_sentence_re.sub(
            r"\1 \2", text
        )  # пробел после знака препинания
        text = self._lower_upper_glue_re.sub(
            " ", text
        )  # склейка нижний/верхний регистр
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text.strip()
