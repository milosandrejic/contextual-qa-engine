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
- [x] Add retrieval tuning: top_k sweep benchmark → default top_k=3 (metadata filtering, MMR, chunk size still open)

---

# Part C — Modern Retrieval Pipeline

> **Goal:** turn the current single-retriever RAG into a production-grade pipeline:
> query understanding → hybrid retrieval → fusion → rerank → eval-driven tuning.
> Every phase is gated by metrics from Phase 6 — no tuning without measurement.

## Phase 6 — Evaluation Harness (foundation)

- [x] Build golden set: 20–30 `(question, expected_answer, expected_source_chunks)` in `benchmark/`
- [x] Retrieval metrics: Recall@K, MRR, nDCG@10
- [x] End-to-end metrics via RAGAS: faithfulness, answer relevance, context precision
- [x] `scripts/eval_retrieval.py` runner that takes a retriever + golden set → JSON report
- [x] Persist results to `benchmark/results/` with timestamp for run-to-run diffs
- [x] Document how to read LangSmith traces for cost + latency

## Phase 7 — Hybrid Search (BM25 + semantic)

- [x] Migrate vectors from Chroma to **pgvector** (single-store architecture)
  - [x] Switch Postgres image to `pgvector/pgvector:pg16` in `docker-compose.yml`
  - [x] Add `pgvector` Python package to `requirements.txt`
  - [x] Alembic migration: `CREATE EXTENSION IF NOT EXISTS vector;`
  - [x] Create `Chunk` SQLAlchemy model: `id`, `document_id` (FK), `source`, `page`, `chunk_index`, `content`, `embedding vector(1536)`, `metadata JSONB`, `created_at`
  - [x] Alembic migration: `chunks` table + HNSW index on `embedding` (cosine) + btree indexes on `source`, `document_id`
  - [x] New service `app/services/pg_vector_store.py`: `store_chunks`, `search_chunks` (pgvector cosine via `<=>`), `delete_by_source`
  - [x] Add `VECTOR_BACKEND` env flag (`chroma | pgvector`); wire `/ask`, `/search`, `/upload`, `/documents` to selected backend
  - [x] Re-run `scripts/eval_retrieval.py` against pgvector; confirm parity (recall@5 ≥ 0.60, nDCG@10 ≥ 0.74)
  - [x] Remove Chroma code + dependency after parity confirmed
- [x] `LexicalRetriever` (Postgres FTS via `websearch_to_tsquery` + `ts_rank_cd`)
- [x] `SemanticRetriever` (pgvector cosine)
- [ ] **RRF fusion** as a pure, unit-tested function
- [ ] `RETRIEVAL_MODE` env var: `semantic | lexical | hybrid` for A/B
- [ ] Benchmark all three modes; document where each wins

## Phase 8 — Reranking

- [ ] Add cross-encoder reranker (`bge-reranker-v2-m3` local **or** Cohere Rerank)
- [ ] Pipeline: hybrid top 50 → rerank → top 5
- [ ] Re-run benchmark, compare nDCG@10 with/without rerank
- [ ] Note bi-encoder vs cross-encoder tradeoffs in `RESOURCES.md`

## Phase 9 — Query Understanding

- [ ] LLM **query rewriting** (history-aware, for conversational `/ask`)
- [ ] **HyDE** toggle (hypothetical document embeddings)
- [ ] **Multi-query expansion** (3 paraphrases → union before fusion)
- [ ] Metadata filter extraction from natural language (e.g. dates, doc types)
- [ ] Benchmark each variant individually

## Phase 10 — Indexing Improvements

- [ ] **Contextual chunking** (Anthropic-style): prepend LLM-generated chunk context before embedding
- [ ] **Parent-document / small-to-big**: embed small chunks, return parent at retrieval (`parent_id` column)
- [ ] Sweep chunk size (200 / 400 / 800) and overlap (0% / 10% / 20%) via benchmark

## Phase 11 — Parameter Calibration

- [ ] Sweep runner: hybrid α (or RRF k), top-K per retriever, rerank N, chunk size
- [ ] Output CSV/JSON to `benchmark/results/sweeps/`
- [ ] Pareto frontier report: quality vs latency vs cost
- [ ] Pick + commit defaults backed by numbers

## Phase 12 — Observability & Feedback Loop

- [ ] Per-stage latency & cost logging (retrieve / rerank / generate) in `messages` table
- [ ] `/feedback` endpoint (thumbs up/down) → store on message row
- [ ] Failing-query → golden-set feedback loop (auto-promote thumbs-down to eval set)

---

# Part D — Production Hardening

## Phase 13 — Production Hardening

- [ ] Rate limiting & input validation
- [ ] Background job processing for document ingestion (FastAPI BackgroundTasks)
