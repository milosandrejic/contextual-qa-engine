from langchain_text_splitters import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    source: str,
    chunk_size: int = 500,
    overlap: int = 50,
    page: int | None = None,
) -> list[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
    )

    split_chunks = splitter.split_text(text)

    return [
        {
            "text": content,
            "metadata": {
                "source": source,
                "chunk_index": index,
                "page": page,
            },
        }
        for index, content in enumerate(split_chunks)
        if content.strip()
    ]
