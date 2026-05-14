from fastapi import APIRouter
from pydantic import BaseModel
from app.services.pg_vector_store import search_chunks

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_documents(request: SearchRequest):
    results = await search_chunks(query=request.query, top_k=request.top_k)

    return {
        "query": request.query,
        "results": results,
    }
