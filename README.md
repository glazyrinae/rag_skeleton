# RAG Skeleton

Небольшой локальный RAG-сервис на FastAPI для индексации файлов проекта и поиска по ним через эмбеддинги. Репозиторий в текущем состоянии использует:

- `FastAPI` для HTTP API
- `Deep Lake` как векторное хранилище
- `FastEmbed` для генерации эмбеддингов
- `Ollama` как LLM backend
- `Docker Compose` для локального запуска

README ниже описывает именно текущее состояние репозитория, без устаревших модулей и интеграций.

## Что умеет сервис

- индексирует каталоги с файлами `.py`, `.md`, `.txt`, `.pdf`, `.html`
- режет документы на чанки через `SentenceSplitter`
- сохраняет тексты, метаданные и эмбеддинги в отдельные датасеты Deep Lake
- ищет релевантные фрагменты по запросу
- отправляет произвольный prompt в Ollama через API `/ask`
- выполняет простой RAG через API `/rag`

Важно: `/ask` по-прежнему не использует найденный контекст из векторной базы автоматически. Для этого теперь есть отдельный endpoint `/rag`.

## Структура проекта

```text
.
├── app/
│   ├── api/endpoints.py        # HTTP endpoints
│   ├── core/db.py              # индексация, чанкинг, эмбеддинги, поиск
│   ├── core/llm.py             # клиент Ollama
│   ├── services/dependencies.py
│   └── main.py                 # точка входа FastAPI
├── deploy/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── data/                   # данные для индексации
│   ├── deeplake_data/          # датасеты Deep Lake
│   ├── embedding_models/       # кэш embedding-моделей
│   └── ollama_models/          # модели Ollama
├── .env.example
├── docker-compose.yml
└── README.md
```

## API

### `POST /add`

Индексирует локальный каталог в указанный датасет.

Пример:

```bash
curl -X POST "http://localhost:5005/add?schema=core&project_path=/app/data/core"
```

Параметры:

- `schema` - имя датасета Deep Lake
- `project_path` - путь внутри контейнера до каталога с документами

### `POST /search`

Ищет похожие чанки и возвращает их как plain text.

Пример:

```bash
curl -X POST "http://localhost:5005/search?schema=core&query=как работает индексация"
```

### `POST /ask`

Отправляет вопрос напрямую в Ollama и возвращает текст ответа.

Примеры:

```bash
curl -X POST "http://localhost:5005/ask?question=Объясни назначение этого сервиса"
```

```bash
curl -X POST "http://localhost:5005/ask" \
  -H "Content-Type: application/json" \
  -d '{"question":"Сформулируй краткое описание проекта"}'
```

### `POST /rag`

Ванильный RAG: ищет релевантные чанки в Deep Lake, собирает prompt с контекстом и отправляет его в Ollama.

Примеры:

```bash
curl -X POST "http://localhost:5005/rag?schema=core&question=Как работает индексация"
```

```bash
curl -X POST "http://localhost:5005/rag" \
  -H "Content-Type: application/json" \
  -d '{
    "schema": "core",
    "question": "Как сервис режет документы на чанки?",
    "top_k": 5,
    "include_sources": true
  }'
```

Ответ возвращается в JSON и содержит:

- `answer` - ответ модели
- `schema` - датасет, в котором выполнялся поиск
- `question` - исходный вопрос
- `used_chunks` - сколько чанков попало в контекст
- `sources` - найденные фрагменты и их метаданные

## Как работает индексация

1. `SimpleDirectoryReader` обходит каталог рекурсивно.
2. Загружаются только файлы с поддерживаемыми расширениями.
3. Документы режутся на чанки через `SentenceSplitter`.
4. Для чанков строятся эмбеддинги моделью `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`.
5. Чанки записываются в Deep Lake датасет вида `/app/deeplake_data/<schema>`.

Параметры по умолчанию находятся в [app/core/db.py](/home/andrey/projects/pets/rag_skeleton/app/core/db.py):

- размер чанка: `1000`
- overlap: `150`
- размер батча эмбеддингов: `64`

## Быстрый старт через Docker Compose

### 1. Подготовьте внешнюю docker-сеть

`docker-compose.yml` ожидает внешнюю сеть `shared_network`.

```bash
docker network create shared_network
```

### 2. Создайте `.env`

Проще всего взять шаблон:

```bash
cp .env.example .env
```

Минимальный пример:

```env
UID=1000
GID=1000
LOG_LEVEL=INFO
OLLAMA_BASE_URL=http://ollama:11438
OLLAMA_MODEL=hodza/cotype-nano-1.5-unofficial:latest
```

Если сервис `ollama` публикуется на другом порту, укажите также:

```env
OLLAMA_PUBLISHED_PORT=11438
```

### 3. Поднимите сервисы

```bash
docker compose up -d --build
```

После старта будут доступны:

- API: `http://localhost:5005`
- Swagger UI: `http://localhost:5005/docs`
- Ollama: `http://localhost:11438`

### 4. Скачайте модель в сервис `ollama`

Сначала убедитесь, что контейнер `ollama` запущен:

```bash
docker compose up -d ollama
```

После этого скачайте нужную модель прямо через сервис:

```bash
docker compose exec ollama ollama pull hodza/cotype-nano-1.5-unofficial:latest
```

Пример для другой модели:

```bash
docker compose exec ollama ollama pull qwen2.5:3b
```

Проверьте, что модель появилась:

```bash
docker compose exec ollama ollama list
```

Важно:

- имя в `OLLAMA_MODEL` в `.env` должно совпадать с реально скачанной моделью
- модели сохраняются в `deploy/ollama_models`

### 5. Положите данные для индексации

Скопируйте файлы в каталог:

```text
deploy/data/<schema_name>
```

Например:

```text
deploy/data/core
```

В контейнере этот путь будет доступен как:

```text
/app/data/core
```

### 6. Запустите индексацию

```bash
curl -X POST "http://localhost:5005/add?schema=core&project_path=/app/data/core"
```

### 7. Выполните поиск

```bash
curl -X POST "http://localhost:5005/search?schema=core&query=FastAPI endpoint"
```

### 8. Выполните RAG-запрос

```bash
curl -X POST "http://localhost:5005/rag?schema=core&question=Как работает индексация"
```

## Локальный запуск без Docker

Требуется Python `3.10`.

```bash
python3.10 -m venv .venv
source .venv/bin/activate
pip install -r deploy/requirements.txt
```

Запуск API:

```bash
cd app
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

При локальном запуске нужно отдельно обеспечить:

- доступный Ollama
- директории для `embedding_models` и `deeplake_data`
- корректный `OLLAMA_BASE_URL`

## Переменные окружения

Основные переменные:

- `OLLAMA_BASE_URL` - адрес Ollama API, по умолчанию `http://ollama:11438`
- `OLLAMA_MODEL` - модель Ollama для `/ask` и `/rag`
- `OLLAMA_PUBLISHED_PORT` - порт, который пробрасывается наружу для сервиса `ollama`
- `LOG_LEVEL` - уровень логирования
- `UID` и `GID` - пользователь контейнера в Docker Compose
- `DEEPLAKE_*` - необязательные переменные для локальной настройки Deep Lake

## Хранилища и volume'ы

Используются следующие монтирования:

- `./deploy/data:/app/data`
- `./app:/app`
- `./deploy/embedding_models:/app/embedding_models`
- `./deploy/deeplake_data:/app/deeplake_data`
- `./deploy/ollama_models:/data/models`

Это значит:

- исходные документы живут в `deploy/data`
- индекс Deep Lake сохраняется в `deploy/deeplake_data`
- скачанные embedding-модели кэшируются в `deploy/embedding_models`
- скачанные Ollama-модели сохраняются в `deploy/ollama_models`

## Ограничения текущей версии

- нет OpenAI-совместимого chat endpoint
- нет Qdrant в текущей реализации
- README старых версий проекта больше не соответствует коду
- для первой загрузки embedding-модели и Ollama-модели нужен доступ в сеть

## Частые проблемы

### Нет прав на запись в `deploy/embedding_models` или `deploy/deeplake_data`

Проверьте владельца каталогов и значения `UID`/`GID` в `.env`.

### Ollama недоступен

Проверьте переменную `OLLAMA_BASE_URL` и убедитесь, что контейнер `ollama` поднят:

```bash
docker compose ps
```

Если контейнер поднят, но модель не найдена, проверьте список моделей:

```bash
docker compose exec ollama ollama list
```

Если нужной модели нет, скачайте её:

```bash
docker compose exec ollama ollama pull <model_name>
```

### Индексация ничего не находит

Проверьте:

- что `project_path` существует внутри контейнера
- что в каталоге есть файлы поддерживаемых типов
- что поиск выполняется по тому же `schema`, куда делалась индексация

## Что имеет смысл сделать дальше

- связать `/search` и `/ask` в единый RAG endpoint
- вернуть structured JSON-ответ для поиска вместо plain text
- добавить управление `k` и другими параметрами поиска через API
- покрыть сервис smoke-тестами
