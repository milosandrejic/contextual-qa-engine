import asyncio
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.services import chat_history
from app.services.vector_store import search_chunks
from app.services.prompt import build_context
from app.services.llm import generate_answer
from app.services.query_builder import build_history_aware_query

router = APIRouter()

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    session_id: uuid.UUID | None = None

@router.post("/ask")
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)):
    """Ask a question with optional session context and message persistence.
    
    Retrieves session history if session_id provided, performs vector search with
    history-aware query, generates answer with LLM, saves conversation to database.
    
    Args:
        request: AskRequest with question, top_k, and optional session_id.
        db: AsyncSession for database operations.
    
    Returns:
        Dict with question, session_id, answer, sources (with citation numbers and metadata),
        and token usage.
    
    Raises:
        HTTPException: 404 if session_id provided but not found.
    """
    history: list[dict] = []

    started_at = time.perf_counter()

    if request.session_id:
        session = await chat_history.get_session(db, request.session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        history = await chat_history.get_session_history(
            db=db,
            session_id=request.session_id,
            limit=settings.max_history_messages,
        )

    retrieval_query = build_history_aware_query(request.question, history)

    chunks = await asyncio.to_thread(search_chunks, query=retrieval_query, top_k=request.top_k)
    context = build_context(chunks)

    result = await asyncio.to_thread(generate_answer, context, request.question, history)
    
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    sources = [
        {
            "citation": index,
            "text": chunk["text"],
            "source": chunk["metadata"].get("source"),
            "page": chunk["metadata"].get("page"),
            "chunk_index": chunk["metadata"].get("chunk_index"),
            "distance": chunk["distance"],
        }
        for index, chunk in enumerate(chunks, start=1)
    ]

    if request.session_id:
        await chat_history.add_message(
            db=db,
            session_id=request.session_id,
            role="user",
            content=request.question,
        )

        await chat_history.add_message(
            db=db,
            session_id=request.session_id,
            role="assistant",
            content=result["answer"],
            sources=sources,
            token_usage=result["usage"],
            latency_ms=latency_ms,
        )

    return {
        "question": request.question,
        "session_id": request.session_id,
        "answer": result["answer"],
        "latency_ms": latency_ms,
        "sources": sources,
        "usage": result["usage"],
    }
