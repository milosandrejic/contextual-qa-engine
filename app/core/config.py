from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    app_name: str = "Contextual QA Engine"
    debug: bool = False

    openai_api_key: str = ""
    openai_model: str = "gpt-4o-mini"
    openai_embedding_model: str = "text-embedding-3-small"

    chroma_api_key: str = ""
    chroma_tenant: str = ""
    chroma_database: str = ""

    database_url: str = ""
    max_history_messages: int = 20
    cors_allow_origins: list[str] = ["http://localhost:3000", "http://localhost:5173"]
    langsmith_tracing: bool = False
    langsmith_api_key: str | None = None
    langsmith_project: str = "contextual-qa-engine"
    langsmith_endpoint: str = "https://api.smith.langchain.com"


settings = Settings()
