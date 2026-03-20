import os, logging
from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, Body
from fastapi.responses import PlainTextResponse
from services.dependencies import get_llm, get_db, get_rag
from core.llm import Llm
from core.db import VectorDB
from core.rag import RAG


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
    k: int = 5,
):
    """Поиск в векторной базе данных"""
    try:
        documents = db.search(query=query, dataset_name=schema, k=k)
        separator = "\n\n" + "=" * 70 + "\n\n"
        output = separator.join(doc.get("text", "") for doc in documents if doc.get("text"))
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


@router.post("/rag", tags=["Работа с RAG"])
async def ask_rag(
    request: Optional[Dict[str, Any]] = Body(default=None),
    question: str = Query("", description="Вопрос пользователя"),
    schema: str = Query("core", description="Имя датасета Deep Lake"),
    top_k: int = Query(5, ge=1, le=20, description="Сколько чанков брать в контекст"),
    include_sources: bool = Query(
        True, description="Возвращать ли найденные источники в ответе"
    ),
    rag: RAG = Depends(get_rag),
):
    """Ванильный RAG: поиск контекста + prompt в LLM."""
    try:
        prompt = question or (request or {}).get("question") or (request or {}).get(
            "prompt", ""
        )
        schema_name = (request or {}).get("schema") or schema
        request_top_k = (request or {}).get("top_k") or top_k
        request_include_sources = (request or {}).get("include_sources")
        if request_include_sources is None:
            request_include_sources = include_sources

        if not prompt:
            raise HTTPException(status_code=400, detail="Не передан question или prompt")

        return rag.ask(
            question=prompt,
            schema=schema_name,
            top_k=request_top_k,
            include_sources=request_include_sources,
        )

    except HTTPException:
        raise
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.exception("RAG request failed schema=%s", schema)
        raise HTTPException(status_code=500, detail=str(e))
