SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use the context below to answer the question.
- If the context does not contain enough information to answer, say "I don't know based on the provided documents."
- Do not make up information.
- Keep answers concise and relevant.
- If you quote from the context, mention the source.

Context:
{context}"""


def build_context(chunks: list[dict]) -> str:
    parts = []

    for i, chunk in enumerate(chunks):
        source = chunk["metadata"].get("source", "unknown")
        page = chunk["metadata"].get("page")
        header = f"[Source: {source}"
        if page:
            header += f", Page {page}"
        header += "]"
        parts.append(f"{header}\n{chunk['text']}")

    return "\n\n---\n\n".join(parts)


def build_messages(context: str, question: str) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT.format(context=context)},
        {"role": "user", "content": question},
    ]
