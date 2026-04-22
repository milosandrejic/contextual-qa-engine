import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.session import Session
from app.models.message import Message


async def create_session(db: AsyncSession) -> Session:
    """Create a new conversation session.
    
    Args:
        db: AsyncSession for database operations.
    
    Returns:
        Newly created Session object with auto-generated ID and timestamps.
    """
    session = Session()

    db.add(session)

    await db.commit()
    await db.refresh(session)

    return session


async def get_sessions(db: AsyncSession) -> list[Session]:
    """Fetch all sessions ordered by creation date (newest first).
    
    Args:
        db: AsyncSession for database operations.
    
    Returns:
        List of all Session objects, ordered descending by created_at.
    """
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    return list(result.scalars().all())

async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    """Fetch a single session with all its messages eagerly loaded.
    
    Args:
        db: AsyncSession for database operations.
        session_id: UUID of the session to fetch.
    
    Returns:
        Session object with messages, or None if not found.
    """
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectinload(Session.messages))
    )

    return result.scalar_one_or_none()

async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
    """Delete a session and all associated messages.
    
    Args:
        db: AsyncSession for database operations.
        session_id: UUID of the session to delete.
    
    Returns:
        True if deleted, False if session not found.
    """
    session = await db.get(Session, session_id)

    if not session:
        return False

    await db.delete(session)
    await db.commit()

    return True

async def add_message(
    db: AsyncSession,
    session_id: uuid.UUID,
    role: str,
    content: str,
    sources: list[dict] | dict | None = None,
    token_usage: dict | None = None,
) -> Message:
    """Add a message to a session.
    
    Args:
        db: AsyncSession for database operations.
        session_id: UUID of the session to add message to.
        role: Message role ('user' or 'assistant').
        content: Message text content.
        sources: Optional list/dict of source citations.
        token_usage: Optional dict with token usage metrics.
    
    Returns:
        Newly created Message object with auto-generated ID and timestamps.
    """
    message = Message(
        session_id=session_id,
        role=role,
        content=content,
        sources=sources,
        token_usage=token_usage,
    )

    db.add(message)
    
    await db.commit()
    await db.refresh(message)

    return message

async def get_session_history(
    db: AsyncSession,
    session_id: uuid.UUID,
    limit: int | None = None,
) -> list[dict]:
    """Fetch recent messages from a session in chronological order.
    
    Fetches the most recent N messages efficiently (DESC + LIMIT), then restores
    chronological order for safe prompt replay. Returns only role and content.
    
    Args:
        db: AsyncSession for database operations.
        session_id: UUID of the session.
        limit: Max number of messages to return. If None or <=0, returns all.
    
    Returns:
        List of message dicts with 'role' and 'content' keys, oldest first.
    """
    query = (
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
    )

    if limit is not None and limit > 0:
        query = query.limit(limit)

    result = await db.execute(query)
    messages_desc = result.scalars().all()
    
    messages = list(reversed(messages_desc))

    return [{"role": msg.role, "content": msg.content} for msg in messages]
