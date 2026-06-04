from app.services.types import HistoryMessage


def build_history_aware_query(
    question: str,
    history: list[HistoryMessage],
    max_messages: int = 6,
) -> str:
    if not history:
        return question

    recent_history = history[-max_messages:]
    history_lines: list[str] = []

    for message in recent_history:
        role = message["role"]
        content = message["content"]

        if role in {"user", "assistant"} and content:
            history_lines.append(f"{role}: {content}")

    if not history_lines:
        return question

    formatted_history = "\n".join(history_lines)

    instruction = "Use this conversation context to resolve references in the current question:"
    retrieval_query = f"{instruction}\n\n{formatted_history}\ncurrent_question: {question}"

    return retrieval_query