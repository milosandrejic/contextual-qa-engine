import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services import chat_history

router = APIRouter(prefix="/sessions", tags=["sessions"])


@router.post("")
async def create_session(db: AsyncSession = Depends(get_db)):
    session = await chat_history.create_session(db)
    return {"session_id": session.id, "created_at": session.created_at}


@router.get("")
async def list_sessions(db: AsyncSession = Depends(get_db)):
    sessions = await chat_history.get_sessions(db)
    return [{"session_id": s.id, "created_at": s.created_at} for s in sessions]


@router.get("/{session_id}/history")
async def get_session_history(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    session = await chat_history.get_session(db, session_id)

    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {
        "session_id": session.id,
        "created_at": session.created_at,
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "sources": m.sources,
                "token_usage": m.token_usage,
                "created_at": m.created_at,
            }
            for m in session.messages
        ],
    }


@router.delete("/{session_id}")
async def delete_session(session_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await chat_history.delete_session(db, session_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"detail": "Session deleted"}
