import asyncio
import uuid
from datetime import datetime, timezone
import sqlalchemy as sa
from app.core.database import async_session
from app.models.chunk import Chunk as ChunkModel
from app.services.embedding import get_embedding, get_embeddings
from app.services.types import Chunk, ChunkInput, ChunkKey, Retriever

# A Chunk is a dict with shape: {text, metadata: {source, page, chunk_index}, score}
# `score` is always higher = better. Its scale depends on the retriever:
#   - semantic: 0..1 (1 = identical, derived from cosine distance)
#   - lexical:  raw ts_rank_cd (unbounded, only meaningful within one result list)
#   - rrf:      raw RRF score (unbounded, only meaningful within one result list)
#   - rerank:   0..1 (Cohere relevance_score, calibrated across queries)

PG_EMBED_BATCH_SIZE = 500


async def store_chunks(chunks: list[ChunkInput], document_id: uuid.UUID) -> int:
    """Embed and insert chunks into the postgres chunks table.

    Embeddings are generated in batches to avoid hitting the OpenAI rate limit.
    Returns the number of rows inserted.
    """
    texts = [c["text"] for c in chunks]
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), PG_EMBED_BATCH_SIZE):
        embeddings.extend(get_embeddings(texts[i : i + PG_EMBED_BATCH_SIZE]))

    rows: list[ChunkModel] = []

    for chunk, embedding in zip(chunks, embeddings):
        metadata = chunk["metadata"]
        filtered_metadata = {k: v for k, v in metadata.items() if v is not None}

        row = ChunkModel(
            id=uuid.uuid4(),
            document_id=document_id,
            source=metadata["source"] or "",
            page=metadata["page"],
            chunk_index=metadata["chunk_index"],
            content=chunk["text"],
            embedding=embedding,
            metadata_=filtered_metadata,
            created_at=datetime.now(timezone.utc),
        )
        rows.append(row)

    async with async_session() as session:
        session.add_all(rows)
        await session.commit()
        # Populate the tsvector column for lexical search.
        await session.execute(
            sa.text(
                "UPDATE chunks SET content_tsv = to_tsvector('english', content) "
                "WHERE content_tsv IS NULL"
            )
        )
        await session.commit()

    return len(rows)


async def search_chunks(query: str, top_k: int = 5) -> list[Chunk]:
    """Return top-k chunks by cosine similarity to the query embedding.

    pgvector's `<=>` operator returns cosine distance in [0, 2].
    We convert to a score in [0, 1] (higher = better) by `1 - distance / 2`.
    """
    query_embedding = get_embedding(query)

    sql = sa.text(
        """
        SELECT
            content,
            source,
            page,
            chunk_index,
            embedding <=> CAST(:embedding AS vector) AS distance
        FROM chunks
        ORDER BY distance
        LIMIT :top_k
        """
    )

    async with async_session() as session:
        result = await session.execute(
            sql,
            {"embedding": str(query_embedding), "top_k": top_k},
        )
        rows = result.fetchall()

    chunks: list[Chunk] = []

    for row in rows:
        chunks.append({
            "text": row.content,
            "metadata": {
                "source": row.source,
                "page": row.page,
                "chunk_index": row.chunk_index,
            },
            "score": 1.0 - float(row.distance) / 2.0,
        })

    return chunks


async def search_chunks_lexical(query: str, top_k: int = 5) -> list[Chunk]:
    """Return top-k chunks by full-text search rank (BM25-style).

    Uses websearch_to_tsquery for natural query parsing (handles quoted phrases,
    OR, -, etc.) and ts_rank_cd for cover-density ranking.
    Score is the raw ts_rank_cd value (higher = better). Unbounded; only
    meaningful for ordering within this result list.
    """
    sql = sa.text(
        """
        SELECT
            content,
            source,
            page,
            chunk_index,
            ts_rank_cd(content_tsv, query) AS rank
        FROM chunks, websearch_to_tsquery('english', :query) query
        WHERE content_tsv @@ query
        ORDER BY rank DESC
        LIMIT :top_k
        """
    )

    async with async_session() as session:
        result = await session.execute(sql, {"query": query, "top_k": top_k})
        rows = result.fetchall()

    chunks: list[Chunk] = []

    for row in rows:
        chunks.append({
            "text": row.content,
            "metadata": {
                "source": row.source,
                "page": row.page,
                "chunk_index": row.chunk_index,
            },
            "score": float(row.rank),
        })

    return chunks


RRF_K = 60


def _chunk_key(chunk: Chunk) -> ChunkKey:
    """Stable identity key for a chunk: (source, page, chunk_index)."""
    metadata = chunk["metadata"]
    return (metadata["source"], metadata["page"], metadata["chunk_index"])


def rrf_fuse(
    results_lists: list[list[Chunk]],
    weights: list[float] | None = None,
    k: int = RRF_K,
) -> list[Chunk]:
    """Fuse ranked result lists with Reciprocal Rank Fusion.

    Score per chunk: sum(w_i / (k + rank_i)) across every retriever that returned it.
    Rank-based, so per-retriever score scales don't need normalisation.
    Returns chunks sorted by descending RRF score; score is the raw RRF sum
    (higher = better, unbounded, only meaningful within this list).
    """
    if weights is None:
        weights = [1.0] * len(results_lists)

    rrf_scores: dict[ChunkKey, float] = {}
    seen_chunks: dict[ChunkKey, Chunk] = {}

    for result_list, retriever_weight in zip(results_lists, weights):
        for rank, chunk in enumerate(result_list, start=1):
            key = _chunk_key(chunk)
            previous_score = rrf_scores.get(key, 0.0)

            rank_contribution = retriever_weight / (k + rank)
            rrf_scores[key] = previous_score + rank_contribution

            if key not in seen_chunks:
                seen_chunks[key] = chunk

    ranked_keys = sorted(rrf_scores, key=rrf_scores.__getitem__, reverse=True)

    fused: list[Chunk] = []

    for key in ranked_keys:
        original = seen_chunks[key]
        fused.append({
            "text": original["text"],
            "metadata": original["metadata"],
            "score": rrf_scores[key],
        })

    return fused


async def search_chunks_hybrid(query: str, top_k: int = 5) -> list[Chunk]:
    """Return top-k chunks using RRF fusion of semantic + lexical results.

    Each retriever fetches 2*top_k candidates so that chunks present in only
    one retriever still have a fair chance of reaching the final top-k.
    """
    semantic, lexical = await asyncio.gather(
        search_chunks(query=query, top_k=top_k * 2),
        search_chunks_lexical(query=query, top_k=top_k * 2),
    )

    return rrf_fuse([semantic, lexical], weights=[1.0, 1.0])[:top_k]


_RETRIEVERS: dict[str, Retriever] = {
    "semantic": search_chunks,
    "lexical": search_chunks_lexical,
    "hybrid": search_chunks_hybrid,
}


def get_retriever(mode: str) -> Retriever:
    """Return the retriever callable for the given mode, defaulting to hybrid."""
    return _RETRIEVERS.get(mode, search_chunks_hybrid)


async def delete_chunks_by_source(source: str) -> int:
    """Delete all chunks whose source matches the given filename.

    Returns the number of rows deleted.
    """
    sql = sa.text("DELETE FROM chunks WHERE source = :source")

    async with async_session() as session:
        result = await session.execute(sql, {"source": source})
        await session.commit()

    return int(result.rowcount)  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType, reportUnknownArgumentType]
