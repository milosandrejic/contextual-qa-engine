# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
# LangChain's public API leaks `Any`/unknown types through prompt templates,
# Runnable chains, and BaseMessage.content. Relax these checks for this file only.

import os
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI
from pydantic import SecretStr
from app.core.config import settings
from app.services.types import HistoryMessage, LLMResult, TokenUsage


def _configure_langsmith_tracing() -> None:
    """Enable LangSmith tracing through environment variables when configured."""
    api_key = settings.langsmith_api_key

    if not (settings.langsmith_tracing and api_key):
        return

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = api_key
    os.environ["LANGSMITH_PROJECT"] = settings.langsmith_project
    os.environ["LANGSMITH_ENDPOINT"] = settings.langsmith_endpoint


_configure_langsmith_tracing()

chat_model = ChatOpenAI(
    api_key=SecretStr(settings.openai_api_key),
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

def _build_chat_history_messages(
    history: list[HistoryMessage] | None,
) -> list[HumanMessage | AIMessage]:
    """Convert raw history messages from database to LangChain message objects.
    
    Args:
        history: List of HistoryMessage entries, or None.
    
    Returns:
        List of HumanMessage or AIMessage objects, empty if history is None.
    """
    if not history:
        return []

    messages: list[HumanMessage | AIMessage] = []

    for message in history:
        role = message["role"]
        content = message["content"]

        if not content:
            continue

        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))

    return messages


def generate_answer(
    context: str,
    question: str,
    history: list[HistoryMessage] | None = None,
) -> LLMResult:
    """Generate an answer using LangChain's RAG chain with chat history.
    
    Invokes the QA chain with context, question, and formatted history.
    Extracts token usage metrics from response if available.
    
    Args:
        context: Formatted context string with citations for the LLM.
        question: User's question to answer.
        history: Optional list of prior history messages for conversation context.
    
    Returns:
        LLMResult with the answer string and token usage metrics.
    """
    response = qa_chain.invoke({
        "context": context,
        "question": question,
        "history": _build_chat_history_messages(history),
    })

    raw_content = response.content
    assert isinstance(raw_content, str), "LangChain chat models must return string content"
    answer = raw_content

    prompt_tokens = 0
    completion_tokens = 0
    total_tokens = 0

    usage_metadata = getattr(response, "usage_metadata", None)

    if usage_metadata:
        prompt_tokens = int(usage_metadata.get("input_tokens", 0))
        completion_tokens = int(usage_metadata.get("output_tokens", 0))
        total_tokens = int(usage_metadata.get("total_tokens", 0))

    usage: TokenUsage = {
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }

    return {
        "answer": answer,
        "usage": usage,
    }
