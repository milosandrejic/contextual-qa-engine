from fastapi import APIRouter
from pydantic import BaseModel
from app.services.vector_store import search_chunks
from app.services.prompt import build_context, build_messages
from app.services.llm import chat

router = APIRouter()


class AskRequest(BaseModel):
    question: str
    top_k: int = 5


@router.post("/ask")
def ask_question(request: AskRequest):
    chunks = search_chunks(query=request.question, top_k=request.top_k)
    context = build_context(chunks)
    messages = build_messages(context=context, question=request.question)
    result = chat(messages)

    sources = [
        {
            "source": chunk["metadata"].get("source"),
            "page": chunk["metadata"].get("page"),
            "chunk_index": chunk["metadata"].get("chunk_index"),
            "distance": chunk["distance"],
        }
        for chunk in chunks
    ]

    return {
        "question": request.question,
        "answer": result["answer"],
        "sources": sources,
        "usage": result["usage"],
    }
