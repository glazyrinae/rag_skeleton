import logging
import os
from typing import Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, ConfigDict, Field

from services.dependencies import get_rag_by_db

logger = logging.getLogger(__name__)
router = APIRouter()


def _normalize_response_text(text: str) -> str:
    """Convert escaped control sequences to printable formatting for chat UIs."""
    return text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "\t")


class AddIndexRequest(BaseModel):
    db_name: str
    index_type: Literal["vector", "tree", "kg", "bm25"]
    folder_name: str
    overwrite: bool = False


class AskRequest(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    session_id: str | None = Field(None, alias="id_session")
    db_name: str
    index_type: Literal["vector", "tree", "kg", "bm25", "hybrid"] = "vector"
    question: str


@router.post("/add-index", tags=["Работа с Векторной БД"])
async def add_index(
    payload: AddIndexRequest,
):
    data_root = os.getenv("RAG_DATA_ROOT", "/app/data")
    folder_path = os.path.join(data_root, payload.folder_name)
    if not payload.db_name.strip():
        raise HTTPException(status_code=400, detail="db_name не должен быть пустым")

    if not os.path.isdir(folder_path):
        raise HTTPException(
            status_code=404,
            detail=f"Папка не найдена: {folder_path}",
        )

    rag_service = get_rag_by_db(payload.db_name, read_only=False)
    try:
        rag_service.add_index(payload.index_type, folder_path, payload.overwrite)
    except Exception as exc:
        logger.exception("Не удалось построить индекс type=%s", payload.index_type)
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    return {
        "status": "success",
        "db_name": payload.db_name,
        "index_type": payload.index_type,
        "folder_name": payload.folder_name,
        "folder_path": folder_path,
        "overwrite": payload.overwrite,
    }


@router.post("/ask", tags=["Работа с RAG"], response_class=PlainTextResponse)
async def ask(
    payload: AskRequest,
):
    if not payload.question.strip():
        raise HTTPException(status_code=400, detail="question не должен быть пустым")
    if not payload.db_name.strip():
        raise HTTPException(status_code=400, detail="db_name не должен быть пустым")

    session_id = (payload.session_id or "").strip()
    rag_service = get_rag_by_db(payload.db_name, read_only=True)
    try:
        answer = rag_service.chat(
            text=payload.question,
            session_id=session_id,
            index_type=payload.index_type,
        )
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Ошибка RAG ask session_id=%s", payload.session_id)
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return _normalize_response_text(answer)
    # return {
    #     "session_id": session_id or None,
    #     "db_name": payload.db_name,
    #     "question": payload.question,
    #     "answer": answer,
    #     "index_type": payload.index_type,
    # }
