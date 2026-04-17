import chromadb
from app.core.config import settings

COLLECTION_NAME = "documents"

client = chromadb.CloudClient(
    api_key=settings.chroma_api_key,
    tenant=settings.chroma_tenant,
    database=settings.chroma_database,
)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
