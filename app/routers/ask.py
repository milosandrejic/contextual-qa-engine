import asyncio
import uuid
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.core.config import settings
from app.services import chat_history
from app.services.vector_store import search_chunks
from app.services.prompt import build_context, build_messages
from app.services.llm import chat

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5
    session_id: uuid.UUID | None = None


@router.post("/ask")
async def ask_question(request: AskRequest, db: AsyncSession = Depends(get_db)):
    history: list[dict] = []
    
    if request.session_id:
        session = await chat_history.get_session(db, request.session_id)

        if not session:
            raise HTTPException(status_code=404, detail="Session not found")

        history = await chat_history.get_session_history(
            db=db,
            session_id=request.session_id,
            limit=settings.max_history_messages,
        )

    chunks = await asyncio.to_thread(search_chunks, query=request.question, top_k=request.top_k)
    context = build_context(chunks)
    messages = build_messages(context=context, question=request.question, history=history)
    result = await asyncio.to_thread(chat, messages)

    sources = [
        {
            "source": chunk["metadata"].get("source"),
            "page": chunk["metadata"].get("page"),
            "chunk_index": chunk["metadata"].get("chunk_index"),
            "distance": chunk["distance"],
        }
        for chunk in chunks
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
        )

    return {
        "question": request.question,
        "session_id": request.session_id,
        "answer": result["answer"],
        "sources": sources,
        "usage": result["usage"],
    }
