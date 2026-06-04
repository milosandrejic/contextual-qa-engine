import asyncio
import time
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.services import chat_history
from app.services.pg_vector_store import get_retriever
from app.services.reranker import rerank_chunks
from app.services.prompt import build_context
from app.services.llm import generate_answer
from app.services.query_builder import build_history_aware_query
from app.services.types import HistoryMessage, Source
from app.routers.schemas import AskResponse

router = APIRouter()

class AskRequest(BaseModel):
    question: str
    top_k: int = 3
    session_id: uuid.UUID | None = None

@router.post("/ask", response_model=AskResponse)
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)) -> AskResponse:
    """Ask a question with optional session context and message persistence.
    
    Retrieves session history if session_id provided, performs vector search with
    history-aware query, generates answer with LLM, saves conversation to database.
    
    Args:
        request: AskRequest with question, top_k, and optional session_id.
        db: AsyncSession for database operations.
    
    Returns:
        AskResponse with question, session_id, answer, sources, and token usage.
    
    Raises:
        HTTPException: 404 if session_id provided but not found.
    """
    history: list[HistoryMessage] = []

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

    retrieve = get_retriever(settings.retrieval_mode)

    # Fetch a larger candidate pool, then rerank down to top_k.
    # The reranker sees more candidates → better chance of surfacing the right chunk.
    candidates = await retrieve(query=retrieval_query, top_k=settings.reranker_candidate_k)
    chunks = await asyncio.to_thread(rerank_chunks, retrieval_query, candidates, request.top_k)

    context = build_context(chunks)

    result = await asyncio.to_thread(generate_answer, context, request.question, history)
    
    latency_ms = int((time.perf_counter() - started_at) * 1000)

    sources: list[Source] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        sources.append({
            "citation": index,
            "text": chunk["text"],
            "source": metadata["source"],
            "page": metadata["page"],
            "chunk_index": metadata["chunk_index"],
            "relevance": round(chunk["score"] * 100),
        })

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

    return AskResponse(
        question=request.question,
        session_id=request.session_id,
        answer=result["answer"],
        latency_ms=latency_ms,
        sources=sources,
        usage=result["usage"],
    )
