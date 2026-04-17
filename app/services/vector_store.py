import chromadb

CHROMA_DIR = "chroma_data"
COLLECTION_NAME = "documents"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(name=COLLECTION_NAME)
