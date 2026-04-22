SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use the context below to answer the question.
- Use the prior conversation messages (if provided) to understand follow-up questions and references like "it", "that", "they", or "this".
- If the context does not contain enough information to answer, say "I don't know based on the provided documents."
- Do not make up information.
- Keep answers concise and relevant.
- If you quote from the context, mention the source.

Context:
{context}"""


def build_context(chunks: list[dict]) -> str:
    formatted_chunks: list[str] = []

    for chunk in chunks:
        metadata = chunk["metadata"]
        source_name = metadata.get("source", "unknown")
        page_number = metadata.get("page")

        context_header = f"[Source: {source_name}"

        if page_number:
            context_header += f", Page {page_number}"

        context_header += "]"

        chunk_text = chunk["text"]
        formatted_chunks.append(f"{context_header}\n{chunk_text}")

    return "\n\n---\n\n".join(formatted_chunks)


def build_messages(context: str, question: str, history: list[dict] | None = None) -> list[dict]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]

    if history:
        # Keep only valid conversational roles when replaying prior turns.
        messages.extend(msg for msg in history if msg.get("role") in {"user", "assistant"})

    messages.append({"role": "user", "content": question})

    return messages
