"""Shared TypedDicts for service-layer data shapes.

These types describe the dict shapes that flow between services, retrievers,
the LLM, and the API layer. Define once here, reuse everywhere — no bare
`dict` annotations in service signatures.
"""

from typing import Protocol, TypedDict


class ChunkMetadata(TypedDict):
    source: str
    page: int | None
    chunk_index: int


class ChunkInput(TypedDict):
    """Chunk before retrieval scoring (output of chunker, input to vector store)."""
    text: str
    metadata: ChunkMetadata


class Chunk(TypedDict):
    """Chunk returned by retrievers and reranker, with relevance score attached."""
    text: str
    metadata: ChunkMetadata
    score: float


class PdfPage(TypedDict):
    text: str
    page: int


class HistoryMessage(TypedDict):
    role: str
    content: str


class TokenUsage(TypedDict):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMResult(TypedDict):
    answer: str
    usage: TokenUsage


class Source(TypedDict):
    citation: int
    text: str
    source: str | None
    page: int | None
    chunk_index: int | None
    relevance: int


ChunkKey = tuple[str | None, int | None, int | None]


class Retriever(Protocol):
    async def __call__(self, query: str, top_k: int = 5) -> list[Chunk]: ...
