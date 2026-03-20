import os
import json
import requests
import logging
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)


class Llm:
    """Простой класс для работы с Ollama"""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        timeout: int = 60,
    ):
        """
        Инициализация клиента Ollama

        Args:
            base_url: URL сервера Ollama (например http://ollama:11434)
            model: Название модели по умолчанию
            timeout: Таймаут запроса в секундах
        """
        self.base_url = (base_url or os.getenv("OLLAMA_BASE_URL", "http://ollama:11438")).rstrip("/")
        self.model = model or os.getenv("OLLAMA_MODEL", "llama3")
        self.timeout = timeout

        # Проверяем доступность сервера
        self._check_health()

    def _check_health(self) -> bool:
        """Проверка доступности сервера"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=5)
            if response.status_code == 200:
                models = response.json().get("models", [])
                logger.info(
                    f"✅ Подключено к Ollama. Доступные модели: {[m['name'] for m in models]}"
                )
                return True
            else:
                logger.warning(f"⚠️ Ollama ответил с кодом {response.status_code}")
                return False
        except Exception as e:
            logger.warning(f"⚠️ Не удалось подключиться к Ollama: {e}")
            return False

    def ask(
        self,
        prompt: str,
        temperature: float = 0.7,
        # max_tokens: Optional[int] = None,
        stream: bool = False,
    ) -> str:
        """
        Отправить запрос к модели

        Args:
            prompt: Текст запроса
            model: Модель для использования (если None, используется модель по умолчанию)
            system: Системный промпт
            temperature: Температура (0.0 - 1.0)
            max_tokens: Максимальное количество токенов
            stream: Потоковый вывод (если True, возвращает генератор)

        Returns:
            Ответ модели
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "stream": stream,
        }

        # if system:
        #     payload["system"] = system

        # if max_tokens:
        #     payload["max_tokens"] = max_tokens

        try:
            logger.debug(f"📤 Отправка запроса к {payload['model']}")

            if stream:
                return self._stream_response(url, payload)
            else:
                return self._simple_response(url, payload)

        except Exception as e:
            logger.error(f"❌ Ошибка при запросе к Ollama: {e}")
            raise

    def _simple_response(self, url: str, payload: Dict) -> str:
        """Получение простого ответа"""
        response = requests.post(url, json=payload, timeout=self.timeout)

        if response.status_code != 200:
            raise Exception(
                f"Ollama вернул ошибку: {response.status_code} - {response.text}"
            )

        result = response.json()
        return result.get("response", "")

    def _stream_response(self, url: str, payload: Dict):
        """Потоковый ответ (генератор)"""
        with requests.post(
            url, json=payload, stream=True, timeout=self.timeout
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Ollama вернул ошибку: {response.status_code}")

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    yield chunk.get("response", "")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: Optional[str] = None,
        temperature: float = 0.7,
        stream: bool = False,
    ) -> str:
        """
        Чат с историей сообщений

        Args:
            messages: Список сообщений [{"role": "user", "content": "..."}, ...]
            model: Модель для использования
            temperature: Температура
            stream: Потоковый вывод

        Returns:
            Ответ модели
        """
        url = f"{self.base_url}/api/chat"

        payload = {
            "model": model or self.model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
        }

        try:
            if stream:
                return self._stream_chat_response(url, payload)
            else:
                response = requests.post(url, json=payload, timeout=self.timeout)

                if response.status_code != 200:
                    raise Exception(f"Ollama вернул ошибку: {response.status_code}")

                result = response.json()
                return result.get("message", {}).get("content", "")

        except Exception as e:
            logger.error(f"❌ Ошибка при чате с Ollama: {e}")
            raise

    def _stream_chat_response(self, url: str, payload: Dict):
        """Потоковый чат"""
        with requests.post(
            url, json=payload, stream=True, timeout=self.timeout
        ) as response:
            if response.status_code != 200:
                raise Exception(f"Ollama вернул ошибку: {response.status_code}")

            for line in response.iter_lines():
                if line:
                    chunk = json.loads(line)
                    if "message" in chunk:
                        yield chunk["message"].get("content", "")

    def list_models(self) -> List[str]:
        """Получить список доступных моделей"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)

            if response.status_code == 200:
                models = response.json().get("models", [])
                return [m["name"] for m in models]
            else:
                return []

        except Exception as e:
            logger.error(f"❌ Ошибка при получении списка моделей: {e}")
            return []
