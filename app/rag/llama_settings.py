import os

from llama_index.core import Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


def configure_llama_settings() -> None:
    """Configure global LlamaIndex Settings for embedding and LLM."""
    Settings.embed_model = OllamaEmbedding(
        model_name=os.getenv("OLLAMA_EMBED_MODEL", "bge-m3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11438"),
        embed_batch_size=int(os.getenv("OLLAMA_EMBED_BATCH_SIZE", "32")),
        ollama_additional_kwargs={
            "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "8")),
        },
    )
    Settings.llm = Ollama(
        model=os.getenv("OLLAMA_MODEL", "llama3"),
        base_url=os.getenv("OLLAMA_BASE_URL", "http://ollama:11438"),
        temperature=0.1,
        request_timeout=float(os.getenv("OLLAMA_REQUEST_TIMEOUT", "300")),
        context_window=int(os.getenv("OLLAMA_CONTEXT_WINDOW", "16384")),
        additional_kwargs={
            "num_thread": int(os.getenv("OLLAMA_NUM_THREADS", "8")),
        },
    )
