from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
    page: int | None = None,
) -> list[dict]:
    """Split text into overlapping chunks with metadata.
    
    Uses RecursiveCharacterTextSplitter to maintain semantic coherence.
    Filters out empty chunks after splitting.
    
    Args:
        text: Raw text content to split.
        source: Source identifier (filename, URL, etc).
        chunk_size: Target size for each chunk in characters.
        overlap: Number of characters to overlap between chunks.
        page: Optional page number for PDF or multi-page documents.
    
    Returns:
        List of dicts with 'text' and 'metadata' keys.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    split_chunks = splitter.split_text(text)

    chunks: list[dict] = []

    for index, content in enumerate(split_chunks):
        if not content.strip():
            continue

        chunk_dict = {
            "text": content,
            "metadata": {
                "source": source,
                "chunk_index": index,
                "page": page,
            },
        }
        chunks.append(chunk_dict)

    return chunks
