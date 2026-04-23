# Plan

> **Approach:** Build everything manually first (raw Python + OpenAI API) to understand
> what happens under the hood. Then upgrade to LangChain/LangGraph once the concepts are solid.

---

# Part A — Manual RAG (Raw Python)

## Phase 0 — Project Setup

- [x] Initialize Python project
- [x] Set up project structure (routers, services, models, core)
- [x] Configure environment variables with `pydantic-settings`
- [x] Set up FastAPI app with health check endpoint
- [x] Add `.env.example` with required keys (OpenAI, etc.)
- [x] Add `.gitignore`
- [x] Dockerize from the start (`Dockerfile` + `docker-compose.yml`)
- [x] Add `Makefile` with common commands (up, down, logs, etc.)

## Phase 1 — Document Ingestion (Manual)

- [x] Build document loader for `.txt` files (plain Python file reading)
- [x] Add PDF loader with `pypdf` (extract text page by page)
- [x] Implement chunking manually (fixed-size with overlap, split on sentences)
- [x] Understand why chunk size & overlap matter (experiment with values)
- [x] Add chunk metadata (source file, chunk index, page number)
- [x] Create `/upload` endpoint that accepts files
- [x] Handle bad uploads (corrupt PDFs, empty files, unsupported formats)
- [x] Store raw files locally (later: S3/cloud)

## Phase 2 — Embeddings & Vector Store (Manual)

- [x] Call OpenAI embeddings API directly (`openai.embeddings.create`)
- [x] Understand what embeddings are and what the response looks like
- [x] Set up ChromaDB as local vector store
- [x] Store chunks + embeddings + metadata in Chroma manually
- [x] Store embeddings as vectors and understand dimensionality
- [x] Build retrieval service: embed query → cosine similarity → top-k chunks
- [x] Create `/search` endpoint to test retrieval standalone

## Phase 3 — RAG Pipeline (Manual MVP)

- [x] Design system prompt template with `{context}` and `{question}` placeholders
- [x] Add "I don't know" handling (instruct LLM in prompt to say so when context is insufficient)
- [x] Call OpenAI chat completions API directly with injected context
- [x] Measure token usage per request
- [x] Build the full pipeline manually: retrieve → build prompt → call LLM
- [x] Create `/ask` endpoint (return answer + source documents in response)
- [x] Test end-to-end: upload doc → ask question → get answer

## Phase 4 — Conversation & UX

### 4a — Database Setup
- [x] Add PostgreSQL service to `docker-compose.yml` + persistent volume
- [x] Add `sqlalchemy[asyncio]`, `asyncpg`, `alembic` to requirements
- [x] Add `DATABASE_URL` to config + `.env.example`
- [x] Create `app/core/database.py` (async engine, session factory, `get_db` dependency)
- [x] Create SQLAlchemy models: `Session` + `Message` (with JSONB sources/usage)
- [x] Init Alembic, configure for async, generate first migration

### 4b — Session & History Endpoints
- [x] Create session management endpoints (`POST /sessions`, `GET /sessions`, `GET /sessions/{id}/history`, `DELETE /sessions/{id}`)
- [x] Update `/ask` to accept optional `session_id` — load history, save Q&A to DB
- [x] Convert `ask.py` to async (for async DB sessions)

### 4c — Conversation Quality
- [x] Prepend conversation history to prompt (1follow-up questions, resolve "it", "that", etc.)
- [x] Limit conversation history (sliding window — last N messages)
- [x] Add citation system (LLM uses [1], [2] format, return numbered sources)

---

# Part B — LangChain / LangGraph Upgrade

## Phase 5 — Migrate to LangChain

- [x] Replace manual document loaders with LangChain loaders
- [x] Replace manual chunking with `RecursiveCharacterTextSplitter`
- [x] Replace manual embeddings + Chroma with LangChain's `Chroma` vectorstore
- [x] Replace manual RAG pipeline with LangChain's chain (retriever → prompt → LLM)
- [x] Compare: what LangChain adds vs. what you built manually
- [x] Integrate LangSmith for tracing & debugging chains
- [ ] Add retrieval tuning: metadata filtering, MMR, chunk size/overlap config
- [ ] Add advanced retrieval: reranking, hybrid search (BM25 + vector), query rewriting

## Phase 6 — Evaluation & Observability

- [ ] Log all queries, retrieved chunks, and responses
- [ ] Add evaluation with RAGAS (faithfulness, relevance, context precision)
- [ ] Build a small golden dataset (20-30 Q&A pairs) for regression testing
- [ ] Add latency tracking per pipeline step
- [ ] Use LangSmith traces to debug bad answers
- [ ] Cost monitoring (LLM + embeddings usage)
- [ ] Prompt versioning

## Phase 7 — Agent Layer (LangGraph)

- [ ] Build retrieval agent with LangGraph (decides if/how to retrieve)
- [ ] Add tool-calling: `search_docs`, `get_metadata`, `summarize_doc`
- [ ] Multi-step QA agent (plan → retrieve → answer → verify)
- [ ] Retry / fallback logic when retrieval quality is low

## Phase 8 — Production Hardening

- [ ] Add Redis for caching frequent queries / embeddings
- [ ] Rate limiting & input validation
- [ ] Migrate to Pinecone or Weaviate (managed vector DB)
- [ ] Add background job processing for document ingestion

---

## Notes

- **Part A is the learning foundation.** Every concept (chunking, embeddings, retrieval, prompt building) is done manually so you understand exactly what LangChain abstracts away.
- **Part B replaces your manual code** with LangChain equivalents. By then you'll immediately understand what each LangChain component does because you've already built it yourself.
- **LangGraph for agents** — explicit state machine control, much better than the older AgentExecutor pattern.
- **RAGAS for evaluation** — structured metrics (faithfulness, answer relevance, context precision) out of the box.
- **Hybrid search (BM25 + vector)** — pure vector search misses exact keyword matches. Combining both catches more.
- **Query rewriting is high-impact** — simple to implement, dramatically improves retrieval for vague questions.
- **Golden dataset** — 20-30 Q&A pairs lets you measure if changes actually improve quality. Build it early.
