import os, logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import PlainTextResponse
from services.dependencies import get_llm, get_db
from core.llm import Llm
from core.db import VectorDB


logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/add", tags=["Работа с векторной БД"])
async def index_local_project(
    db: VectorDB = Depends(get_db),
    schema: str = "core",
    project_path: str = "/app/data",
):
    """Поиск в векторной базе данных"""

    # rag.clear_conversation_history()
    if not os.path.exists(project_path):
        raise HTTPException(status_code=404, detail="Директория проекта не найдена")

    # Индексация проекта
    db.add_docs_to_db(dataset_name=schema, project_path=project_path)

    return {
        "status": "success",
        "schema": schema,
        "project_path": project_path,
    }


@router.post(
    "/search", response_class=PlainTextResponse, tags=["Работа с векторной БД"]
)
async def search(
    query: str,
    db: VectorDB = Depends(get_db),
    schema: str = "core",
):
    """Поиск в векторной базе данных"""
    try:
        documents = db.search(query, schema)
        separator = "\n\n" + "=" * 70 + "\n\n"
        output = separator.join(f"{doc.page_content}" for doc in documents)
        return output

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ask", response_class=PlainTextResponse, tags=["Работа с LLM"])
async def ask_llm(
    request: Optional[Dict[str, Any]] = Body(default=None),
    question: str = Query("", description="Вопрос пользователя"),
    llm: Llm = Depends(get_llm),
):
    """Задать вопрос LLM с использованием RAG"""
    try:
        prompt = question or (request or {}).get("question") or (request or {}).get(
            "prompt", ""
        )
        if not prompt:
            raise HTTPException(status_code=400, detail="Не передан question или prompt")

        logger.info("LLM request started model=%s prompt_len=%s", llm.model, len(prompt))
        res = llm.ask(prompt=prompt)
        logger.info("LLM request finished model=%s response_len=%s", llm.model, len(res))
        return res

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("LLM request failed model=%s", llm.model)
        raise HTTPException(status_code=500, detail=str(e))
