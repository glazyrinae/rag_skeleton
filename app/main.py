import logging
import os
from fastapi import FastAPI
from api.endpoints import router as api_router


def configure_logging() -> None:
    log_level = os.getenv("LOG_LEVEL", "INFO").upper()
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


configure_logging()

app = FastAPI()
app.include_router(api_router, prefix="")
