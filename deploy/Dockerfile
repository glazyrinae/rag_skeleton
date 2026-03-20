# Dockerfile
FROM python:3.10-slim

WORKDIR /app

# Установка системных зависимостей
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    git \
    && rm -rf /var/lib/apt/lists/*

# Копирование requirements
COPY requirements.txt .
# Установка Python зависимостей
RUN pip install --no-cache-dir -r requirements.txt

# Preload NLTK assets into a writable runtime path.
RUN mkdir -p /tmp/nltk_data \
    && python -c "import nltk; [nltk.download(pkg, download_dir='/tmp/nltk_data') for pkg in ('stopwords', 'punkt', 'punkt_tab')]"
ENV NLTK_DATA=/tmp/nltk_data
