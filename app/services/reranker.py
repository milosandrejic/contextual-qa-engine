"""Cohere cross-encoder reranker.

Pipeline position: hybrid retrieval (top candidate_k) → rerank → top_k → LLM

Why a cross-encoder improves on bi-encoder retrieval:
  - Bi-encoders (our pgvector semantic search) embed query and document
    independently, then compare vectors. Fast, but the two representations
    never "see" each other during encoding.
  - Cross-encoders take the (query, document) pair as a single input and
    produce a relevance score. Slower, but far more accurate because the
    model attends over both texts jointly.

We use Cohere's hosted rerank-v3-5 model to avoid running a large model
locally. The trade-off: ~300–500ms network latency per /ask request, but
zero GPU requirement and no model download.
"""

import cohere
from app.core.config import settings

# Module-level client — initialised once, reused across requests.
_client: cohere.Client | None = None


def _get_client() -> cohere.Client:
    global _client
    if _client is None:
        if not settings.cohere_api_key:
            raise RuntimeError(
                "COHERE_API_KEY is not set. Add it to .env to use the reranker."
            )
        _client = cohere.Client(api_key=settings.cohere_api_key)
    return _client


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """Rerank a candidate pool of chunks using Cohere's cross-encoder.

    Args:
        query:   The user's question (or history-aware query) — same string
                 used for retrieval so the reranker scores against the same intent.
        chunks:  Candidate chunks from retrieval (typically 50).
                 Each chunk has {text, metadata: {source, page, chunk_index}, score}.
        top_k:   Number of chunks to return after reranking.

    Returns:
        top_k chunks sorted by descending Cohere relevance score.
        The ``score`` field is set to Cohere's raw ``relevance_score`` (0..1,
        higher = better, calibrated across queries).
    """
    if not chunks:
        return chunks

    client = _get_client()

    response = client.rerank(
        model=settings.reranker_model,
        query=query,
        documents=[chunk["text"] for chunk in chunks],
        top_n=top_k,
    )

    return [
        {
            **chunks[result.index],
            "score": result.relevance_score,
        }
        for result in response.results
    ]
