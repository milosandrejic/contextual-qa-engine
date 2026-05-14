from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings
from app.services.pg_vector_store import get_retriever

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_documents(request: SearchRequest):
    retrieve = get_retriever(settings.retrieval_mode)
    results = await retrieve(query=request.query, top_k=request.top_k)

    return {
        "query": request.query,
        "results": results,
    }
