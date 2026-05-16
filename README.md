# contextual-qa-engine

A production-style RAG (Retrieval-Augmented Generation) backend built with FastAPI.
Upload documents, ask questions, get grounded answers with citations.

The pipeline combines **hybrid search** (semantic + lexical) with **cross-encoder
reranking** for high-quality retrieval, and persists conversations in Postgres so
follow-up questions can use prior context.

---

## Features

- **Document ingestion** — PDF and text files, chunked with metadata (source, page, chunk index)
- **Hybrid retrieval** — pgvector (HNSW) + Postgres FTS (GIN) fused with Reciprocal Rank Fusion
- **Cross-encoder reranking** — Cohere `rerank-english-v3.0` re-orders the candidate pool
- **Conversational `/ask`** — sessions, sliding-window history, numbered citations `[1] [2]`
- **Pluggable retrieval mode** — switch between `semantic | lexical | hybrid` via env var
- **Benchmark harness** — golden set + Recall@K / MRR / nDCG@10 metrics
- **LangSmith tracing** — optional, for cost/latency observability

## Architecture

```
upload ─▶ chunker ─▶ embeddings ─▶ pgvector (vector + tsvector columns)
                                          │
ask ─▶ query rewrite ─▶ hybrid retrieve (RRF) ─▶ Cohere rerank ─▶ LLM ─▶ answer + citations
                                                                    │
                                                              session history (Postgres)
```

**Stack**: FastAPI · SQLAlchemy (async) · Alembic · pgvector · OpenAI (`gpt-4o-mini`,
`text-embedding-3-small`) · Cohere · LangChain (loaders + chunkers only).

## Quick start

```bash
# 1. Configure secrets
cp .env.example .env
# edit .env: set OPENAI_API_KEY and (optional) COHERE_API_KEY

# 2. Start the stack (FastAPI + Postgres with pgvector)
make up

# 3. Apply migrations
make migration-up

# 4. Sanity check
curl http://localhost:8000/health
```

The API is now at `http://localhost:8000`. OpenAPI docs at `/docs`.

## API endpoints

| Method | Path | Purpose |
|---|---|---|
| `POST` | `/upload` | Upload a PDF or text file, ingest into the vector store |
| `GET`  | `/documents` | List ingested documents |
| `DELETE` | `/documents/{source}` | Remove a document and its chunks |
| `POST` | `/search` | Raw retrieval — returns top chunks (no LLM call) |
| `POST` | `/ask` | Ask a question. Accepts optional `session_id` for follow-ups |
| `POST` | `/sessions` | Create a new conversation session |
| `GET`  | `/sessions` | List sessions |
| `GET`  | `/sessions/{id}/history` | Get full message history for a session |
| `DELETE` | `/sessions/{id}` | Delete a session |

## Configuration

All settings are env vars (see [.env.example](.env.example)). The important ones:

| Variable | Default | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | — | Required, for LLM + embeddings |
| `COHERE_API_KEY` | — | Required for reranking |
| `DATABASE_URL` | `postgresql+asyncpg://postgres:postgres@db:5432/contextual_qa` | Postgres connection (asyncpg driver) |
| `RETRIEVAL_MODE` | `hybrid` | `semantic` / `lexical` / `hybrid` |
| `RERANKER_MODEL` | `rerank-english-v3.0` | Cohere reranker model |
| `RERANKER_CANDIDATE_K` | `20` | Candidate pool size before reranking |
| `MAX_HISTORY_MESSAGES` | `20` | Sliding window for conversational context |
| `LANGSMITH_TRACING` | `false` | Set `true` to send traces to LangSmith |

## Development

```bash
make up               # start app + postgres
make down             # stop
make logs             # tail container logs
make shell            # exec into the app container
make migration-up     # apply Alembic migrations
make migration-create m="add foo"   # generate new migration
```

For local (non-Docker) script runs, point at the host-mapped Postgres port:

```bash
source .venv/bin/activate
DATABASE_URL=postgresql+asyncpg://postgres:postgres@localhost:5433/contextual_qa \
  python scripts/eval_retrieval.py --top-k 5 --mode hybrid --rerank
```

## Benchmarking

The retrieval pipeline is benchmarked against a 25-question golden set in
[benchmark/golden_set.json](benchmark/golden_set.json). Current best configuration:

| Configuration | recall@5 | MRR | nDCG@10 |
|---|---|---|---|
| semantic only (pgvector) | 0.60 | 0.42 | 0.7334 |
| lexical only (FTS) | 0.16 | 0.12 | 0.1659 |
| hybrid (RRF) | 0.60 | 0.43 | 0.7499 |
| **hybrid + Cohere rerank** | **0.68** | **0.55** | **0.8678** |

Full methodology, formulas, and per-run history are in
[benchmark/BENCHMARK.md](benchmark/BENCHMARK.md).

## Project structure

```
app/
  core/         # config, database engine
  models/       # SQLAlchemy models (Document, Chunk, Session, Message)
  routers/      # FastAPI endpoints
  services/     # chunker, embedding, pg_vector_store, reranker, llm, prompt, ...
alembic/        # database migrations
benchmark/      # golden set, metrics, results
scripts/        # eval_retrieval.py and other CLI tools
data/           # uploaded files + cached chunk dumps (gitignored)
```

## Documentation

- [PLAN.md](PLAN.md) — phased development roadmap, what's done and what's next
- [benchmark/BENCHMARK.md](benchmark/BENCHMARK.md) — retrieval evaluation methodology and results
- [RESOURCES.md](RESOURCES.md) — design notes, sources for test corpora, LangSmith usage
- [CODING_STYLE.md](CODING_STYLE.md) — code conventions used in this repo

