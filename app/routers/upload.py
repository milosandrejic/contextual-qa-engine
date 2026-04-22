import json
import shutil
from pathlib import Path
from fastapi import APIRouter, UploadFile, File, HTTPException
from app.services.document_loader import load_txt, load_pdf
from app.services.chunker import chunk_text
from app.services.vector_store import store_chunks

router = APIRouter()

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_DIR = Path("data/chunks")
CHUNKS_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".txt", ".pdf"}


@router.post("/upload")
def upload_document(file: UploadFile = File(...)):
    """Upload and process a document (.txt or .pdf).
    
    Validates file type, saves to disk, extracts and chunks text, stores in vector DB.
    Returns chunk metadata and a sample of the chunks created.
    
    Args:
        file: Uploaded file object.
    
    Returns:
        Dict with filename, chunk count, vector DB storage status, chunks file path, and sample chunks.
    
    Raises:
        HTTPException: 400 if filename missing, file type unsupported, file empty, or processing fails.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = Path(file.filename).suffix.lower()

    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {ext}. Allowed: {', '.join(ALLOWED_EXTENSIONS)}",
        )

    file_path = UPLOAD_DIR / file.filename

    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

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

    stored = store_chunks(chunks)

    return {
        "filename": file.filename,
        "chunks": len(chunks),
        "stored_in_vector_db": stored,
        "chunks_file": str(chunks_file),
        "sample": chunks[:2] if chunks else [],
    }
