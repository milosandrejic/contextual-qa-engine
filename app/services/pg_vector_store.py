import uuid
from datetime import datetime, timezone
import sqlalchemy as sa
from app.core.database import async_session
from app.models.chunk import Chunk
from app.services.embedding import get_embedding, get_embeddings

PG_EMBED_BATCH_SIZE = 500


async def store_chunks(chunks: list[dict], document_id: uuid.UUID) -> int:
    """Embed and insert chunks into the postgres chunks table.

    Embeddings are generated in batches to avoid hitting the OpenAI rate limit.
    Returns the number of rows inserted.
    """
    texts = [c["text"] for c in chunks]
    embeddings: list[list[float]] = []

    for i in range(0, len(texts), PG_EMBED_BATCH_SIZE):
        embeddings.extend(get_embeddings(texts[i : i + PG_EMBED_BATCH_SIZE]))

    rows = [
        Chunk(
            id=uuid.uuid4(),
            document_id=document_id,
            source=chunk["metadata"].get("source", ""),
            page=chunk["metadata"].get("page"),
            chunk_index=chunk["metadata"].get("chunk_index", idx),
            content=chunk["text"],
            embedding=embedding,
            metadata_={k: v for k, v in chunk["metadata"].items() if v is not None},
            created_at=datetime.now(timezone.utc),
        )
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings))
    ]

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


async def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks by cosine similarity to the query embedding."""
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

    return [
        {
            "text": row.content,
            "metadata": {
                "source": row.source,
                "page": row.page,
                "chunk_index": row.chunk_index,
            },
            "distance": float(row.distance),
        }
        for row in rows
    ]


async def search_chunks_lexical(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks by full-text search rank (BM25-style).

    Uses websearch_to_tsquery for natural query parsing (handles quoted phrases,
    OR, -, etc.) and ts_rank_cd for cover-density ranking.
    Distance is returned as 1 - rank so lower = better (consistent with cosine).
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

    return [
        {
            "text": row.content,
            "metadata": {
                "source": row.source,
                "page": row.page,
                "chunk_index": row.chunk_index,
            },
            "distance": 1.0 - float(row.rank),
        }
        for row in rows
    ]


RRF_K = 60


def _chunk_key(chunk: dict) -> tuple:
    m = chunk["metadata"]
    return (m.get("source"), m.get("page"), m.get("chunk_index"))


def rrf_fuse(results_lists: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """Fuse ranked result lists with Reciprocal Rank Fusion.

    Score per chunk: sum(1 / (k + rank_i)) across every retriever that returned it.
    Rank-based, so cosine distances and ts_rank_cd values need no normalisation.
    Returns chunks sorted by descending RRF score; distance = 1/score (lower = better).
    """
    scores: dict[tuple, float] = {}
    chunks_by_key: dict[tuple, dict] = {}

    for result_list in results_lists:
        for rank, chunk in enumerate(result_list, start=1):
            key = _chunk_key(chunk)
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in chunks_by_key:
                chunks_by_key[key] = chunk

    fused = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)

    return [
        {**chunks_by_key[key], "distance": 1.0 / scores[key]}
        for key in fused
    ]


async def search_chunks_hybrid(query: str, top_k: int = 5) -> list[dict]:
    """Return top-k chunks using RRF fusion of semantic + lexical results.

    Each retriever fetches 2*top_k candidates so that chunks present in only
    one retriever still have a fair chance of reaching the final top-k.
    """
    import asyncio
    semantic, lexical = await asyncio.gather(
        search_chunks(query=query, top_k=top_k * 2),
        search_chunks_lexical(query=query, top_k=top_k * 2),
    )
    return rrf_fuse([semantic, lexical])[:top_k]


_RETRIEVERS = {
    "semantic": search_chunks,
    "lexical": search_chunks_lexical,
    "hybrid": search_chunks_hybrid,
}


def get_retriever(mode: str):
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

    return result.rowcount
