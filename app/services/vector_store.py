import uuid
import chromadb
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from app.core.config import settings

COLLECTION_NAME = "documents"

client = chromadb.CloudClient(
    api_key=settings.chroma_api_key,
    tenant=settings.chroma_tenant,
    database=settings.chroma_database,
)

collection = client.get_or_create_collection(name=COLLECTION_NAME)

embedding_function = OpenAIEmbeddings(
    api_key=settings.openai_api_key,
    model=settings.openai_embedding_model,
)

vector_store = Chroma(
    client=client,
    collection_name=COLLECTION_NAME,
    embedding_function=embedding_function,
)


def store_chunks(chunks: list[dict]) -> int:
    documents = [
        Document(
            page_content=chunk["text"],
            metadata={k: v for k, v in chunk["metadata"].items() if v is not None},
        )
        for chunk in chunks
    ]
    ids = [str(uuid.uuid4()) for _ in chunks]

    vector_store.add_documents(documents=documents, ids=ids)

    return len(chunks)


def search_chunks(query: str, top_k: int = 5) -> list[dict]:
    results = vector_store.similarity_search_with_score(query=query, k=top_k)

    return [
        {
            "text": document.page_content,
            "metadata": document.metadata,
            "distance": score,
        }
        for document, score in results
    ]


def delete_chunks_by_source(source: str) -> int:
    """Delete all vector chunks whose metadata.source matches the given filename.

    Returns the number of chunks deleted.
    """
    existing = collection.get(where={"source": source})
    ids = existing.get("ids") or []

    if not ids:
        return 0

    collection.delete(ids=ids)

    return len(ids)
