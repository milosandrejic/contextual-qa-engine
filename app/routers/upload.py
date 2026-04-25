import asyncio
import json
import shutil
from pathlib import Path
from fastapi import APIRouter, Depends, UploadFile, File, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.database import get_db
from app.services.document_loader import load_txt, load_pdf
from app.services.chunker import chunk_text
from app.services.vector_store import store_chunks
from app.services import document as document_service

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("/upload")
async def upload_document(
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """Upload and process a document (.txt or .pdf).

    Validates file type, saves to disk, extracts and chunks text, stores chunks
    in the vector DB, and persists a Document row in the relational DB.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    existing = await document_service.get_document_by_filename(db, file.filename)

    if existing:
        raise HTTPException(
            status_code=409,
            detail=f"Document with filename '{file.filename}' already exists.",
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    file_size = file_path.stat().st_size
    page_count: int | None = None

    try:
        if ext == ".txt":
            text = load_txt(str(file_path))

            if not text.strip():
                raise HTTPException(status_code=400, detail="File is empty.")

            chunks = chunk_text(text=text, source=file.filename)

        elif ext == ".pdf":
            pages = load_pdf(str(file_path))

            if not pages:
                raise HTTPException(status_code=400, detail="PDF has no extractable text.")

            page_count = len(pages)
            chunks = []

            for page in pages:
                page_chunks = chunk_text(
                    text=page["text"],
                    source=file.filename,
                    page=page["page"],
                )
                chunks.extend(page_chunks)

    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to process file. It may be corrupt.")

    chunks_file = CHUNKS_DIR / f"{Path(file.filename).stem}.json"

    with open(chunks_file, "w") as f:
        json.dump(chunks, f, indent=2)

    stored = await asyncio.to_thread(store_chunks, chunks)

    document = await document_service.create_document(
        db=db,
        filename=file.filename,
        file_size=file_size,
        chunk_count=len(chunks),
        page_count=page_count,
    )

    return {
        "id": document.id,
        "filename": document.filename,
        "file_size": document.file_size,
        "page_count": document.page_count,
        "chunk_count": document.chunk_count,
        "indexed_at": document.indexed_at,
        "stored_in_vector_db": stored,
        "chunks_file": str(chunks_file),
    }
