from openai import OpenAI
from app.core.config import settings

client = OpenAI(api_key=settings.openai_api_key)

def get_embedding(text: str) -> list[float]:
    response = client.embeddings.create(
        input=text,
        model=settings.openai_embedding_model,
    )

    return response.data[0].embedding

def get_embeddings(texts: list[str]) -> list[list[float]]:
    response = client.embeddings.create(
        input=texts,
        model=settings.openai_embedding_model,
    )

    return [item.embedding for item in response.data]
