import uuid
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services import document

router = APIRouter(prefix="/documents", tags=["documents"])


@router.get("")
async def list_documents(db: AsyncSession = Depends(get_db)):
    documents = await document.get_documents(db)

    return [
        {
            "id": d.id,
            "filename": d.filename,
            "file_size": d.file_size,
            "page_count": d.page_count,
            "chunk_count": d.chunk_count,
            "indexed_at": d.indexed_at,
        }
        for d in documents
    ]


@router.get("/{document_id}")
async def get_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    doc = await document.get_document(db, document_id)

    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    return {
        "id": doc.id,
        "filename": doc.filename,
        "file_size": doc.file_size,
        "page_count": doc.page_count,
        "chunk_count": doc.chunk_count,
        "indexed_at": doc.indexed_at,
    }


@router.delete("/{document_id}")
async def delete_document(document_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    deleted = await document.delete_document(db, document_id)

    if not deleted:
        raise HTTPException(status_code=404, detail="Document not found")

    return {"detail": "Document deleted"}
