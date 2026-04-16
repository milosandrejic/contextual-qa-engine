def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
    page: int | None = None,
) -> list[dict]:
    chunks = []
    start = 0
    index = 0

    while start < len(text):
        end = start + chunk_size
        content = text[start:end].strip()

        if content:
            chunk = {
                "text": content,
                "metadata": {
                    "source": source,
                    "chunk_index": index,
                    "page": page,
                },
            }
            chunks.append(chunk)
            index += 1

        start += chunk_size - overlap

    return chunks
