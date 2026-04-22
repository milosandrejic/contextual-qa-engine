from fastapi import APIRouter
from pydantic import BaseModel
from app.services.vector_store import search_chunks

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
def search_documents(request: SearchRequest):
    """Search for documents in the vector store by semantic similarity.
    
    Args:
        request: SearchRequest with 'query' string and optional 'top_k' (default 5).
    
    Returns:
        Dict with query and results list (each with text, metadata, distance).
    """
    results = search_chunks(query=request.query, top_k=request.top_k)

    return {
        "query": request.query,
        "results": results,
    }
