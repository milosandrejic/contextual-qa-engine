import uuid
from datetime import datetime, timezone
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.document import Document


async def create_document(
    db: AsyncSession,
    filename: str,
    file_size: int,
    chunk_count: int,
    page_count: int | None = None,
) -> Document:
    document = Document(
        filename=filename,
        file_size=file_size,
        chunk_count=chunk_count,
        page_count=page_count,
        indexed_at=datetime.now(timezone.utc),
    )

    db.add(document)

    await db.commit()
    await db.refresh(document)

    return document


async def get_documents(db: AsyncSession) -> list[Document]:
    result = await db.execute(select(Document).order_by(Document.indexed_at.desc()))
    return list(result.scalars().all())


async def get_document(db: AsyncSession, document_id: uuid.UUID) -> Document | None:
    return await db.get(Document, document_id)


async def get_document_by_filename(db: AsyncSession, filename: str) -> Document | None:
    result = await db.execute(select(Document).where(Document.filename == filename))
    return result.scalars().first()


async def delete_document(db: AsyncSession, document_id: uuid.UUID) -> bool:
    document = await db.get(Document, document_id)

    if not document:
        return False

    await db.delete(document)
    await db.commit()

    return True
