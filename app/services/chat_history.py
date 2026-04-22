import uuid
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload
from app.models.session import Session
from app.models.message import Message


async def create_session(db: AsyncSession) -> Session:
    session = Session()

    db.add(session)

    await db.commit()
    await db.refresh(session)

    return session


async def get_sessions(db: AsyncSession) -> list[Session]:
    result = await db.execute(select(Session).order_by(Session.created_at.desc()))
    return list(result.scalars().all())


async def get_session(db: AsyncSession, session_id: uuid.UUID) -> Session | None:
    result = await db.execute(
        select(Session)
        .where(Session.id == session_id)
        .options(selectinload(Session.messages))
    )

    return result.scalar_one_or_none()


async def delete_session(db: AsyncSession, session_id: uuid.UUID) -> bool:
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


async def get_session_history(db: AsyncSession, session_id: uuid.UUID) -> list[dict]:
    result = await db.execute(
        select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.asc())
    )
    
    messages = result.scalars().all()

    return [{"role": msg.role, "content": msg.content} for msg in messages]
