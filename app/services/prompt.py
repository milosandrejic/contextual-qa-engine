SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Rules:
- Only use the context below to answer the question.
- Use the prior conversation messages (if provided) to understand follow-up questions and references like "it", "that", "they", or "this".
- If the context does not contain enough information to answer, say "I don't know based on the provided documents."
- Do not make up information.
- Keep answers concise and relevant.
- Add citations in square brackets that point to context block numbers, for example [1] or [2].
- Cite claims grounded in context using the matching context numbers.
- If multiple blocks support a claim, include multiple citations like [1][3].

Context:
{context}"""


def build_context(chunks: list[dict]) -> str:
    """Format ranked search results into a numbered context string for the LLM.
    
    Each chunk is numbered [1], [2], etc., with source and page info in header.
    Chunks are separated by visual dividers (---) for clarity.
    
    Args:
        chunks: List of dicts with 'text', 'metadata' keys from vector search.
    
    Returns:
        Formatted context string ready for LLM prompts.
    """
    formatted_chunks: list[str] = []

    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        source_name = metadata.get("source", "unknown")
        page_number = metadata.get("page")

        context_header = f"[{index}] [Source: {source_name}"

        if page_number:
            context_header += f", Page {page_number}"

        context_header += "]"

        chunk_text = chunk["text"]
        formatted_chunks.append(f"{context_header}\n{chunk_text}")

    return "\n\n---\n\n".join(formatted_chunks)


def build_messages(context: str, question: str, history: list[dict] | None = None) -> list[dict]:
    """Build a message list for API calls (system + history + user question).
    
    Injects formatted context into system prompt. Replays prior messages only if their
    role is 'user' or 'assistant' (filters out malformed entries).
    
    Args:
        context: Formatted context string from build_context().
        question: User's question to append.
        history: Optional list of prior message dicts with 'role' and 'content'.
    
    Returns:
        List of message dicts ready for LLM APIs.
    """
    messages = [{"role": "system", "content": SYSTEM_PROMPT.format(context=context)}]

    if history:
        valid_messages = [msg for msg in history if msg.get("role") in {"user", "assistant"}]
        messages.extend(valid_messages)

    messages.append({"role": "user", "content": question})

    return messages
