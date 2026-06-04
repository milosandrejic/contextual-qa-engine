"""Pydantic response schemas for API endpoints."""

import uuid
from datetime import datetime
from pydantic import BaseModel
from app.services.types import Source, TokenUsage


class HealthResponse(BaseModel):
    status: str


class DeletedResponse(BaseModel):
    detail: str


class DocumentSummary(BaseModel):
    id: uuid.UUID
    filename: str
    file_size: int
    page_count: int | None
    chunk_count: int
    indexed_at: datetime


class UploadResponse(DocumentSummary):
    stored_in_vector_db: int
    chunks_file: str


class SearchResultItem(BaseModel):
    text: str
    metadata: dict[str, str | int | None]
    relevance: int


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]


class AskResponse(BaseModel):
    question: str
    session_id: uuid.UUID | None
    answer: str
    latency_ms: int
    sources: list[Source]
    usage: TokenUsage


class SessionSummary(BaseModel):
    id: uuid.UUID
    created_at: datetime


class MessageItem(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    sources: list[Source] | None
    token_usage: TokenUsage | None
    latency_ms: int | None
    created_at: datetime


class SessionHistoryResponse(BaseModel):
    id: uuid.UUID
    created_at: datetime
    messages: list[MessageItem]
