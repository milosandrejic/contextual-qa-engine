from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_txt(file_path: str) -> str:
    """Load and extract text from a .txt file.
    
    Args:
        file_path: Absolute path to the .txt file.
    
    Returns:
        Combined text content from all pages, joined with newlines.
    """
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    return "\n".join(doc.page_content for doc in documents)


def load_pdf(file_path: str) -> list[dict]:
    """Load and extract text from a PDF file, page by page.
    
    Skips empty pages. Increments page numbers by 1 (PDFs are 0-indexed).
    
    Args:
        file_path: Absolute path to the PDF file.
    
    Returns:
        List of dicts with 'text' and 'page' keys for non-empty pages.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    pages: list[dict] = []

    for doc in documents:
        text = doc.page_content or ""

        if not text.strip():
            continue

        page_number = int(doc.metadata.get("page", 0)) + 1
        page_dict = {
            "text": text,
            "page": page_number,
        }
        pages.append(page_dict)

    return pages
