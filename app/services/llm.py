from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from app.core.config import settings

chat_model = ChatOpenAI(
    api_key=settings.openai_api_key,
    model=settings.openai_model,
    temperature=0.2,
)

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

qa_prompt_template = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{question}"),
])

qa_chain = qa_prompt_template | chat_model


def _build_chat_history_messages(history: list[dict] | None) -> list[HumanMessage | AIMessage]:
    if not history:
        return []

    messages: list[HumanMessage | AIMessage] = []

    for message in history:
        role = message.get("role")
        content = message.get("content")

        if not content:
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))

        if role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def generate_answer(context: str, question: str, history: list[dict] | None = None) -> dict:
    response = qa_chain.invoke({
        "context": context,
        "question": question,
        "history": _build_chat_history_messages(history),
    })

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    has_usage_metadata = hasattr(response, "usage_metadata") and response.usage_metadata

    if has_usage_metadata:
        prompt_tokens = response.usage_metadata.get("input_tokens", 0)
        completion_tokens = response.usage_metadata.get("output_tokens", 0)
        total_tokens = response.usage_metadata.get("total_tokens", 0)

    return {
        "answer": response.content,
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        },
    }
