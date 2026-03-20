# SSPHERA RAG

Локальная система Retrieval-Augmented Generation, которая индексирует исходный код и документацию, складывает эмбеддинги в Qdrant и позволяет задавать вопросы через API, совместимое с OpenAI. Проект рассчитан на офлайн‑исполнение (всё хранится в ваших каталогах) и может переключаться между локальными LLM (Ollama) и облачными моделями через OpenRouter.

---

## Содержание

1. [Обзор и сценарии применения](#обзор-и-сценарии-применения)
2. [Архитектура и основные компоненты](#архитектура-и-основные-компоненты)
3. [Структура репозитория](#структура-репозитория)
4. [Технологический стек](#технологический-стек)
5. [Подготовка окружения](#подготовка-окружения)
6. [Запуск через Docker Compose](#запуск-через-docker-compose)
7. [Локальный запуск без Docker](#локальный-запуск-без-docker)
8. [Работа с данными и индексацией](#работа-с-данными-и-индексацией)
9. [API](#api)
10. [Настройка эмбеддингов и чанкинга](#настройка-эмбеддингов-и-чанкинга)
11. [Устранение неполадок](#устранение-неполадок)
12. [Дорожная карта и дополнительные модули](#дорожная-карта-и-дополнительные-модули)

---

## Обзор и сценарии применения

- 🔍 **Семантический поиск по коду** — быстрые ответы на вопросы о функциях, моделях данных и шаблонах в репозитории.
- 📚 **Навигация по документации** — индексируются Markdown, PDF и HTML, поэтому можно задавать вопросы к техническим регламентам.
- 🧩 **Смешанные проекты** — поддержка нескольких «схем» (коллекций Qdrant), что позволяет хранить разные проекты в отдельных индексах и выбирать контекст хэштегами (`#core`, `#sber` и т.д.).
- 🤖 **ChatGPT‑совместимый API** — эндпоинт `/v1/chat/completions` понимает формат OpenAI и умеет работать в стриминговом режиме.
- 🛡️ **Полностью локально** — документы и индексы лежат в ваших volume‑каталогах, LLM можно запустить через Ollama.

---

## Архитектура и основные компоненты

```
┌────────────────────┐      ┌────────────────────┐      ┌──────────────────────┐
│ FastAPI (app/api)  │──LLM→│ core.llm.Llm        │──◄──▶ внешняя LLM (Ollama, │
│ /add /search /ask  │      │ - prompt-пайплайн   │      │  OpenRouter)         │
│ /v1/chat/completions│     │ - выбор схемы       │      └──────────────────────┘
│                    │      │ - обращение к VectorDB                            │
└─────────┬──────────┘      └─────────┬──────────┘
          │                           │
          │ индексирует/ищет          │ similarity search
          ▼                           ▼
┌────────────────────┐      ┌──────────────────────────────────────────┐
│ core.db.VectorDB   │──►──▶ Qdrant (docker service `qdrant`)          │
│ - FileProcessor    │      │ коллекции core, sber, test…              │
│ - Recursive splitter│     └──────────────────────────────────────────┘
└────────────────────┘
```

### FastAPI (`app/api/endpoints.py`)
- `/add` — запускает индексацию каталога и складывает документы в выбранную схему.
- `/search` — отдаёт сырые тексты найденных чанков.
- `/ask` и `/v1/chat/completions` — вызывают LLM поверх найденного контекста.

### LLM (`app/core/llm.py`)
- Работает либо с Ollama (`TYPE_MODEL=local`), либо с OpenRouter (`TYPE_MODEL=cloud`).
- Понимает хэштеги в истории чата: `#core`, `#sber`, `#reset`.
- Автоматически строит промпт (правила тональности, ограничение на цитирование).

### VectorDB (`app/core/db.py`)
- Инициализирует коллекции Qdrant на лету, следит за размерностью векторов.
- Поддерживает две стратегии чанкинга: семантическую через Tree‑sitter (класс `PythonTreeSitterSplitter`) и классическую `RecursiveCharacterTextSplitter`. По умолчанию включён второй вариант (чанки ≈1000 символов).
- Использует `HuggingFaceEmbeddings` (`BAAI/bge-m3`, 1024 измерения). Для перехода на GGUF достаточно раскомментировать класс `GGUFEmbeddings`.

### Парсеры файлов (`app/lib/file_processor.py`)
- **Python** — чтение файла целиком + метаданные (путь, тип документа, язык).
- **Markdown** — выделение полного текста, секций, блоков кода.
- **PDF** — извлечение помимо полного текста также страниц.
- **HTML** — очистка тегов и создание документов по блокам.
- Можно легко расширять через `FileProcessorFacade.PARSERS`.

---

## Структура репозитория

```
ssphera_rag/
├── app/
│   ├── api/                # FastAPI роуты и (в будущем) очередь
│   ├── core/               # LLM + VectorDB
│   ├── lib/                # Парсеры и вспомогательные классы
│   ├── services/           # DI (dependencies), адаптеры
│   ├── data/               # Каталог, проброшенный в контейнер как /app/data
│   ├── embedding_models/   # Локальный кэш эмбеддингов
│   ├── main.py             # Инициализация FastAPI
│   └── manage.py           # Хук для очереди (оставлен для обратной совместимости)
├── deploy/
│   ├── Dockerfile, requirements.txt
│   ├── data/               # Примеры проектов (mount → /app/data)
│   ├── embedding_models/   # Область кэша, шарится с контейнером
│   └── qdrant_data/        # Персистентное хранилище Qdrant
├── docker-compose.yml
└── README.md
```

---

## Технологический стек

- **Язык / фреймворки**: Python 3.10, FastAPI, LangChain.
- **LLM-провайдеры**: Ollama (локально) или OpenRouter (cloud, через ChatOpenAI API).
- **Векторная БД**: Qdrant (`qdrant/qdrant:latest`), cosine distance.
- **Эмбеддинги**: `sentence-transformers/BAAI/bge-m3` (по умолчанию), совместимо с GGUF‑моделью `jina-embeddings-v4-text-code`.
- **Парсеры**: Tree‑sitter (Python), Markdown, BeautifulSoup, PyPDF2.
- **Контейнеризация**: Docker Compose, внешний network `shared_network`.

---

## Подготовка окружения

1. **Требования**:
   - Docker + Docker Compose v2.
   - Доступ к внешней сети (для скачивания моделей/зависимостей).
   - 8 ГБ RAM на контейнер API + Qdrant (рекомендация).
2. **Каталоги, которые должны существовать до старта**:
   - `deploy/data` — сюда кладём проекты для индексации (смонтируется в `/app/data`).
   - `deploy/embedding_models` — кэш HuggingFace / GGUF, должен быть доступен на запись.
   - `deploy/qdrant_data` — персистентное хранилище Qdrant.
3. **Переменные окружения** (`.env`):

| Переменная | Назначение |
|-----------|------------|
| `OPENROUTER_API_KEY` | ключ для OpenRouter (используется, если `TYPE_MODEL=cloud`). |
| `TYPE_MODEL` | `cloud` или `local`. В локальном режиме требуется запущенный Ollama и загруженная модель. |
| `QDRANT_HOST` / `QDRANT_PORT` | координаты Qdrant, внутри docker-compose автозамена на `qdrant:6333`. |
| `QDRANT__SERVICE__HTTP_PORT` | публичный порт для доступа к Qdrant UI. |
| `UID` / `GID` | пользователь/группа, от имени которых работают контейнеры (нужно, чтобы иметь права на тома). |

---

## Запуск через Docker Compose

1. Заполните `.env` (см. предыдущий раздел).
2. По желанию раскомментируйте блоки `ollama` и `open-webui` в `docker-compose.yml`.
3. Поднимите сервисы:

```bash
docker compose up -d --build
```

4. Проверьте, что API доступно: `http://localhost:5005/docs`.
5. Убедитесь, что в Qdrant появилась коллекция после первой индексации: `http://localhost:6333/dashboard`.
6. Для просмотра dataset через официальный Activeloop Visualizer поднимите `deeplake-ui` и откройте `http://localhost:8091`.

**Volumes в docker-compose:**
- `./deploy/data:/app/data` — источник документов.
- `./app:/app` — монтируем код для hot-reload (`uvicorn --reload`).
- `./deploy/embedding_models:/app/embedding_models` — общий кэш эмбеддингов.
- `./deploy/qdrant_data:/qdrant/storage` — данные Qdrant.

### Deep Lake Visualizer

В `docker-compose.yml` добавлен сервис `deeplake-ui`, который поднимает локальную страницу-обёртку и встраивает официальный Activeloop Visualizer через `iframe`.

Запуск:

```bash
docker compose up -d deeplake-ui
```

По умолчанию UI доступен на `http://localhost:8091`.

Дополнительные переменные:

| Переменная | Назначение |
|-----------|------------|
| `DEEPLAKE_VISUALIZER_PORT` | Порт локального UI, по умолчанию `8091`. |
| `DEEPLAKE_VISUALIZER_TITLE` | Заголовок страницы, по умолчанию `Deep Lake Viewer`. |
| `DEEPLAKE_VISUALIZER_BASE_URL` | Базовый URL official visualizer, по умолчанию `https://app.activeloop.ai/visualizer`. |
| `DEEPLAKE_VISUALIZER_DATASET_URL` | Dataset URL, который будет открыт сразу после загрузки страницы. |

Ограничение:
- Локальный `deeplake://localhost:6543/...` не открывается в official Activeloop UI напрямую.
- Для просмотра через `deeplake-ui` нужен dataset URL, доступный через Activeloop Visualizer.

---

## SSPHERA MCP (поиск в Qdrant через BAAI/bge-m3) для IDE

Если хочется MCP‑тулзы поверх уже проиндексированных коллекций SSPHERA и HuggingFace‑эмбеддингов `BAAI/bge-m3` из `/app/embedding_models`, поднимите `ssphera_mcp_http`.
`ssphera_mcp_http` реализован на Python и проксирует вызовы в SSPHERA API.

На сервере:

```bash
docker compose --profile mcp up -d --build api qdrant ssphera_mcp_http
```

В Zed (локально):

```json
{
  "context_servers": {
    "ssphera": {
      "url": "http://REMOTE_HOST:8090/mcp",
      "headers": { "X-API-Key": "YOUR_MCP_PROXY_API_KEY" }
    }
  }
}
```

Если IDE не коннектится по `.../mcp`, проверьте вариант `.../sse` (в `mcp-proxy` включены оба транспорта).

### Пример «тулзы» в API: `/tools/review`

Упрощённый code-review эндпоинт (без MCP), который делает RAG‑поиск в Qdrant и просит LLM отревьюить файл/дифф:

```bash
curl -X POST http://localhost:5005/tools/review \
  -H "Content-Type: application/json" \
  -d '{"schema":"core","file_path":"core/llm.py","instructions":"Найди баги и риски"}'
```

---

## Локальный запуск без Docker

1. Установите системные зависимости (Tree‑sitter, gcc/g++).
2. Создайте virtualenv и установите Python-зависимости:

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r deploy/requirements.txt
```

3. Запустите Qdrant отдельно (например, `docker run -p 6333:6333 qdrant/qdrant`).
4. Экспортируйте переменные окружения (`export QDRANT_HOST=localhost`, `TYPE_MODEL=local` и т.д.).
5. Запустите API:

```bash
uvicorn app.main:app --reload --port 8000
```

6. Для локальной LLM поднимите Ollama и скачайте модель:

```bash
ollama pull qwen2.5:0.5b
```

---

## Работа с данными и индексацией

1. Скопируйте репозиторий/документы в папку `deploy/data/<schema_name>` на хосте.
2. В контейнере это окажется в `/app/data/<schema_name>`.
3. Вызовите `/add`:

```bash
curl -X POST "http://localhost:5005/add?schema=test&project_path=/app/data/test"
```

Параметры отправляются как query-string (FastAPI не читает JSON-тело для этих аргументов).

### Как происходит индексация

1. `VectorDB.scan_dataset` обходит каталог, игнорируя `__pycache__`, `.git`, `node_modules`, `venv`.
2. `FileProcessorFacade.parse_file` выбирает подходящий парсер по расширению и создаёт `langchain.schema.Document` с богатыми метаданными (путь, тип элемента, язык, страницы и т.д.).
3. Документы нарезаются `RecursiveCharacterTextSplitter` (длина 1000, overlap 150) и пачками отправляются в Qdrant (`batch_size=100`).
4. Для Python можно переключиться на `PythonTreeSitterSplitter` — он отдаёт чанки «функция / класс целиком» (см. раздел [Настройка эмбеддингов и чанкинга](#настройка-эмбеддингов-и-чанкинга)).

---

## API

| Метод | Описание | Пример |
|-------|----------|--------|
| `POST /add` | Индексация каталога. | `curl -X POST "http://localhost:5005/add?schema=core&project_path=/app/data/core"` |
| `POST /search` | Возвращает `k` самых похожих чанков (по умолчанию `k=5`). Выход — plain text с разделителем. | `curl -X POST "http://localhost:5005/search?schema=core&query=Запрашиваемый%20набор"` |
| `POST /ask` | Упрощённый вызов `Llm.ask`; принимает параметр `user_query` и возвращает ответ LLM. | `curl -X POST "http://localhost:5005/ask" -d "user_query=Что делает отчет #core"` |
| `POST /v1/chat/completions` | OpenAI-совместимый эндпоинт. Поддерживает поле `stream`. | См. ниже |

### Пример вызова `/v1/chat/completions`

```bash
curl -X POST http://localhost:5005/v1/chat/completions \
     -H "Content-Type: application/json" \
     -d '{
           "model": "qwen-local",
           "stream": false,
           "messages": [
             {"role": "user", "content": "Расскажи про структуру проекта #core"}
           ]
         }'
```

Ответ идентичен OpenAI (id, created, choices, usage). В стриминговом режиме сервер отправляет Server-Sent Events (`data: {...}`), что позволяет подключать существующие клиенты ChatGPT.

---

## Настройка эмбеддингов и чанкинга

1. **Выбор модели эмбеддингов** (`app/core/db.py`):
   - HuggingFace (`BAAI/bge-m3`): быстрее стартует, требует интернет/кэш.
   - GGUF (`jina-embeddings-v4-text-code`): полностью офлайн, но нужен `llama-cpp-python` и вес модели в `deploy/embedding_models`.
   - Переключение осуществляется заменой `_init_embeddings` (раскомментируйте блок с `GGUFEmbeddings` и укажите путь к GGUF в `EMBEDDING_MODEL`).

2. **Размер чанков**:
   - Параметры `DEFAULT_CHUNK_SIZE` и `DEFAULT_CHUNK_OVERLAP` управляют `RecursiveCharacterTextSplitter`.
   - Для более точных совпадений по коротким запросам уменьшайте `DEFAULT_CHUNK_SIZE` (например, 600). Не забывайте про overhead на количество чанков и память.

3. **Tree‑sitter сплиттер**:
   - Экземпляр `self.tree_sitter_splitter` уже создаётся, но по умолчанию не используется.
   - Чтобы комбинировать семантическую и фиксированную нарезку, можно заменить
     ```python
     chunks = self.simple_splitter.split_documents(documents)
     ```
     на двухшаговый процесс (Tree‑sitter → рекурсивный splitter для слишком длинных узлов). Такой подход сохраняет метаданные классов/функций и помогает ловить короткие запросы.

4. **Количество результатов в поиске**:
   - Метод `VectorDB.search` принимает параметр `k` (по умолчанию 5). Передайте нужное значение из эндпоинта `/search`, добавив аргумент `k` в обработчик.
   - Если Qdrant отдаёт меньше документов, чем ожидается, проверьте score threshold и наличие данных в коллекции.

---

## Устранение неполадок

| Симптом | Причина | Решение |
|--------|---------|---------|
| `PermissionError: [Errno 13] Permission denied: '/app/embedding_models/...'` | Контейнер API запущен от пользователя `UID:GID`, который не имеет прав записи к примонтированному каталогу. | Выдайте права на `deploy/embedding_models` (`chown -R $USER:$USER deploy/embedding_models`). |
| `/search` всегда возвращает один и тот же нерелевантный чанк | Запрос ищет в другой схеме или коллекция содержит мало документов. | Убедитесь, что параметр `schema` передаётся в URL (`/search?schema=test...`). Перепроверьте, что индексация прошла успешно и коллекция содержит документы. |
| Нужный текст есть в Qdrant, но запрос возвращает другой фрагмент | Чанк содержит много лишнего кода, поэтому короткий текст «теряется» в эмбеддинге. | Уменьшите размер чанка или включите Tree‑sitter. Также можно временно увеличить `k` и фильтровать результаты на клиенте. |
| FastAPI падает при импорте `services.handle_request` | В текущей версии файл отсутствует, но импорт остался (исторический артефакт). | Удалите импорт или создайте заглушку, если нужна совместимость со старыми клиентами. |
| Ничего не индексируется | Проверьте `SUPPORTED_EXTENSIONS` в `app/core/db.py` и убедитесь, что файлы имеют поддерживаемые расширения (`.py`, `.md`, `.txt`, `.pdf`, `.html`). |

---

## Дорожная карта и дополнительные модули

- **Очереди и воркеры**: в `app/manage.py` и `services/rag_adapter.py` оставлены заготовки для фонового воркера, который ранее обрабатывал задачи из Redis (`rag_questions`). Сейчас эти файлы не задействованы; при необходимости можно вернуть очередь, подключив Redis и реализовав функции в `rag_adapter.py`.
- **Telegram-бот**: в README ранних версий упоминался бот (`bot/`), но каталог отсутствует. Если нужен интерфейс мессенджера, придётся реализовать его заново.
- **Open WebUI / Ollama**: блоки уже прописаны в docker-compose и готовы к включению. Нужно лишь раскомментировать службы и убедиться, что внешняя сеть `shared_network` существует.
- **Дополнительные схемы**: расширьте `ALLOWED_SCHEMAS` в `app/core/llm.py`, добавьте соответствующие каталоги в `deploy/data` и вновь проиндексируйте.

---

## Полезные подсказки

- Перед первой индексацией удалите `deploy/qdrant_data`, чтобы избежать конфликтов размерностей при смене модели эмбеддингов.
- В логах FastAPI видно прогресс индексации: найдено документов → создано чанков → батчи.
- В `deploy/data` уже лежит пример проекта `core/sphere_sqla` — удобно для smoke‑тестов.
- Для быстрой отладки используйте Uvicorn вне Docker: `uvicorn app.main:app --reload --port 5005`.

---
