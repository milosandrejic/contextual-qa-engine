from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)


def chat(messages: list[dict]) -> dict:
    response = client.chat.completions.create(
        model=settings.openai_model,
        messages=messages,
        temperature=0.2,
    )

    choice = response.choices[0]

    return {
        "answer": choice.message.content,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
            "total_tokens": response.usage.total_tokens,
        },
    }
