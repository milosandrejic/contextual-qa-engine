import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services import chat_history
from app.routers.schemas import (
    DeletedResponse,
    MessageItem,
    SessionHistoryResponse,
    SessionSummary,
)

router = APIRouter(prefix="/sessions", tags=["sessions"])

@router.post("", response_model=SessionSummary)
async def create_session(db: AsyncSession = Depends(get_db)) -> SessionSummary:
    """Create a new conversation session."""
    session = await chat_history.create_session(db)
    return SessionSummary(id=session.id, created_at=session.created_at)

@router.get("", response_model=list[SessionSummary])
async def list_sessions(db: AsyncSession = Depends(get_db)) -> list[SessionSummary]:
    """List all conversation sessions ordered by creation date (newest first)."""
    sessions = await chat_history.get_sessions(db)
    return [SessionSummary(id=s.id, created_at=s.created_at) for s in sessions]

@router.get("/{session_id}/history", response_model=SessionHistoryResponse)
async def get_session_history(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> SessionHistoryResponse:
    """Fetch full history of a session including all messages and metadata."""
    session = await chat_history.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    messages = [
        MessageItem(
            id=m.id,
            role=m.role,
            content=m.content,
            sources=m.sources,
            token_usage=m.token_usage,
            latency_ms=m.latency_ms,
            created_at=m.created_at,
        )
        for m in session.messages
    ]

    return SessionHistoryResponse(
        id=session.id,
        created_at=session.created_at,
        messages=messages,
    )


@router.delete("/{session_id}", response_model=DeletedResponse)
async def delete_session(
    session_id: uuid.UUID,
    db: AsyncSession = Depends(get_db),
) -> DeletedResponse:
    """Delete a session and all associated messages."""
    deleted = await chat_history.delete_session(db, session_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")

    return DeletedResponse(detail="Session deleted")
