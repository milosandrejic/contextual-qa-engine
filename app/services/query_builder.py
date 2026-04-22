def build_history_aware_query(question: str, history: list[dict], max_messages: int = 6) -> str:
    if not history:
        return question

    recent_history = history[-max_messages:]
    history_lines: list[str] = []

    for message in recent_history:
        role = message.get("role")
        content = message.get("content")

        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return question

    query_parts = [
        "Use this conversation context to resolve references in the current question:",
    ]
    query_parts.extend(history_lines)
    query_parts.append(f"current_question: {question}")

    return "\n".join(query_parts)