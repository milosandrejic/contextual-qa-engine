import uuid
import chromadb
from app.core.config import settings
from app.services.embedding import get_embedding, get_embeddings

COLLECTION_NAME = "documents"

client = chromadb.CloudClient(
    api_key=settings.chroma_api_key,
    tenant=settings.chroma_tenant,
    database=settings.chroma_database,
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)


def store_chunks(chunks: list[dict]) -> int:
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {k: v for k, v in chunk["metadata"].items() if v is not None}
        for chunk in chunks
    ]
    ids = [str(uuid.uuid4()) for _ in chunks]

    embeddings = get_embeddings(texts)

    collection.add(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(chunks)


def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    query_embedding = get_embedding(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
    )

    chunks = []

    for i in range(len(results["ids"][0])):
        chunks.append({
            "text": results["documents"][0][i],
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        })

    return chunks
