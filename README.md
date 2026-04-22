# RAG Skeleton

Локальный RAG-сервис на `FastAPI + Ollama + Deep Lake + Flowise`.

Ниже только рабочий минимум:
1. как запустить систему,
2. как скачать LLM и embedding-модель,
3. как подключить Flowise.

## 1. Быстрый старт

### Требования
- Docker + Docker Compose
- Linux/macOS

### Шаг 1. Подготовить сеть

```bash
docker network create shared_network
docker network create traefik-public
```

Если сеть уже есть, Docker вернет сообщение, что она существует.

### Шаг 2. Подготовить `.env`

```bash
cp .env.example .env
```

Минимально проверьте значения:

```env
UID=1000
GID=1000
LOG_LEVEL=INFO

OLLAMA_BASE_URL=http://ollama:11438
CLOUDE_MODEL=0
OLLAMA_CLOUD_BASE_URL=https://ollama.com
OLLAMA_MODEL=hodza/cotype-nano-1.5-unofficial:latest
OLLAMA_EMBED_MODEL=bge-m3
OLLAMA_PUBLISHED_PORT=11438

FLOWISE_DOMAIN=flowise.example.com
FLOWISE_CORS_ORIGINS=https://site.example.com
FLOWISE_IFRAME_ORIGINS=https://site.example.com
FLOWISE_HTTP_SECURITY_CHECK=true
```

### Шаг 3. Запустить сервисы

Traefik в этом compose **не поднимается**. Он должен быть запущен отдельно в вашем другом проекте и подключен к сети `traefik-public`.

Проверьте DNS: `A`-запись `FLOWISE_DOMAIN` должна указывать на IP сервера, где запущены Docker-контейнеры.

```bash
docker compose up -d --build
```

Проверка:

- API: `http://localhost:5005/docs`
- Flowise: `https://<FLOWISE_DOMAIN>`
- Ollama: `http://localhost:11438`

### Шаг 4. Положить документы

Кладите файлы для индексации в:

```text
deploy/data/<folder_name>
```

Пример:

```text
deploy/data/gost
```

### Шаг 5. Построить индекс

```bash
curl -X POST 'http://localhost:5005/add-index' \
  -H 'Content-Type: application/json' \
  -d '{
    "db_name": "gost",
    "index_type": "bm25",
    "folder_name": "gost",
    "overwrite": true
  }'
```

`index_type`: `vector | tree | kg | bm25`.

### Шаг 6. Задать вопрос

```bash
curl -X POST 'http://localhost:5005/ask' \
  -H 'Content-Type: application/json' \
  -d '{
    "db_name": "gost",
    "index_type": "bm25",
    "question": "О чем этот набор документов?"
  }'
```

Важно: `/ask` принимает **JSON body**. Если передавать `?question=...` в query string, получите `422`.

---

## 2. Как скачать LLM и embedding-модель

В этом проекте обе модели берутся через `Ollama`:
- LLM: `OLLAMA_MODEL`
- Embeddings: `OLLAMA_EMBED_MODEL`

Переключение локальный/облачный Ollama:
- `CLOUDE_MODEL=0` -> использовать `OLLAMA_BASE_URL` (локально, режим по умолчанию)
- `CLOUDE_MODEL=1` -> использовать `OLLAMA_CLOUD_BASE_URL` (облачный endpoint)

### Авторизация для Cloud-моделей

```bash
docker compose exec -it ollama ollama signin
```

### Скачать модели

```bash
docker compose exec ollama ollama pull hodza/cotype-nano-1.5-unofficial:latest
docker compose exec ollama ollama pull bge-m3
```

### Проверить, что модели доступны

```bash
docker compose exec ollama ollama list
```

### Связать имена с приложением

Убедитесь, что `.env` совпадает с тем, что скачали:

```env
OLLAMA_MODEL=hodza/cotype-nano-1.5-unofficial:latest
OLLAMA_EMBED_MODEL=bge-m3
CLOUDE_MODEL=0
OLLAMA_BASE_URL=http://ollama:11438
# OLLAMA_CLOUD_BASE_URL=https://ollama.com
```

После изменения `.env` перезапустите API:

```bash
docker compose up -d api
```

Модели сохраняются в `deploy/ollama_models` (volume из `docker-compose.yml`).

---

## 3. Настройка Flowise

### Базовое подключение

1. Откройте `https://<FLOWISE_DOMAIN>`.
2. Создайте новый `Chatflow` или `Agentflow`.
3. Добавьте HTTP-запрос к вашему API.

### Параметры HTTP-запроса в Flowise

- Method: `POST`
- URL: `http://api:8000/ask`
  - внутри контейнера Flowise используйте `api:8000`, не `localhost:5005`
- Headers:
  - `Content-Type: application/json`
- Body (JSON):

```json
{
  "db_name": "gost",
  "index_type": "bm25",
  "question": "{{question}}"
}
```

Опционально для stateful-чата можно передавать `id_session`:

```json
{
  "id_session": "user-123",
  "db_name": "gost",
  "index_type": "bm25",
  "question": "{{question}}"
}
```

### Вставка чата на другом сайте

Если чат встраивается в другой проект, не используйте `localhost`:

```js
Chatbot.init({
  chatflowid: "YOUR_CHATFLOW_ID",
  apiHost: "https://flowise.example.com"
})
```

Что поменять в проекте с фронтендом:

- `apiHost` заменить с `http://localhost:3000` на `https://<FLOWISE_DOMAIN>`.
- Добавить домен Flowise в CSP (`connect-src` и `frame-src`, если используете iframe).
- Если есть reverse-proxy фронтенда (Nginx/Caddy), разрешить исходящие HTTPS-запросы на Flowise-домен.
- Проверить, что в Flowise выставлены:
  - `FLOWISE_CORS_ORIGINS=https://<домен_вашего_сайта>`
  - `FLOWISE_IFRAME_ORIGINS=https://<домен_вашего_сайта>`

### Типовые ошибки в Flowise

- `422 Unprocessable Entity`
  - Причина: отправка `question` как query param (`/ask?question=...`)
  - Исправление: отправлять JSON body.

- `500 Internal Server Error` с `LockedException` (Deep Lake)
  - Причина: хранилище занято другим процессом записи.
  - Что делать: не запускать параллельные операции записи в один dataset, повторить запрос после завершения индексации/конкурирующей задачи.

---

## Полезные команды

```bash
# Статус контейнеров
docker compose ps

# Логи API
docker logs --tail 200 rag_skeleton-api-1

# Логи Flowise
docker logs --tail 200 rag_skeleton-flowise-1

# Перезапустить только API
docker compose up -d api
```
