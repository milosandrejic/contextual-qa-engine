from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader

def load_txt(file_path: str) -> str:
    path = Path(file_path)
    documents = TextLoader(str(path), encoding="utf-8").load()

    return "\n".join(doc.page_content for doc in documents)


def load_pdf(file_path: str) -> list[dict]:
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pages = []

    for doc in documents:
        text = doc.page_content or ""

        if text.strip():
            pages.append({
                "text": text,
                "page": int(doc.metadata.get("page", 0)) + 1,
            })

    return pages
