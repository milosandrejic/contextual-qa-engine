import asyncio
from fastapi import APIRouter
from pydantic import BaseModel
from app.core.config import settings
from app.services.pg_vector_store import get_retriever
from app.services.reranker import rerank_chunks
from app.routers.schemas import SearchResponse, SearchResultItem

router = APIRouter()


class SearchRequest(BaseModel):
    query: str
    top_k: int = 5


@router.post("/search", response_model=SearchResponse)
async def search_documents(request: SearchRequest) -> SearchResponse:
    retrieve = get_retriever(settings.retrieval_mode)
    candidates = await retrieve(query=request.query, top_k=settings.reranker_candidate_k)
    chunks = await asyncio.to_thread(rerank_chunks, request.query, candidates, request.top_k)

    results: list[SearchResultItem] = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        results.append(SearchResultItem(
            text=chunk["text"],
            metadata={
                "source": metadata["source"],
                "page": metadata["page"],
                "chunk_index": metadata["chunk_index"],
            },
            relevance=round(chunk["score"] * 100),
        ))

    return SearchResponse(query=request.query, results=results)
