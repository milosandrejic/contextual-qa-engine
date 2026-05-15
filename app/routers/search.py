from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings
from app.services.pg_vector_store import get_retriever
from app.services.reranker import rerank_chunks
import asyncio

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search")
async def search_documents(request: SearchRequest):
    retrieve = get_retriever(settings.retrieval_mode)
    candidates = await retrieve(query=request.query, top_k=settings.reranker_candidate_k)
    results = await asyncio.to_thread(rerank_chunks, request.query, candidates, request.top_k)

    return {
        "query": request.query,
        "results": results,
    }
