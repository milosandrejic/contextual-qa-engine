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
- [ ] Store chunks + embeddings + metadata in Chroma manually
- [ ] Store embeddings as vectors and understand dimensionality
- [ ] Build retrieval service: embed query → cosine similarity → top-k chunks
- [ ] Create `/search` endpoint to test retrieval standalone

## Phase 3 — RAG Pipeline (Manual MVP)

- [ ] Build the full pipeline manually: retrieve → build prompt → call LLM
- [ ] Create `/ask` endpoint
- [ ] Design system prompt template with `{context}` and `{question}` placeholders
- [ ] Call OpenAI chat completions API directly with injected context
- [ ] Return answer + source documents in response
- [ ] Add "I don't know" handling (instruct LLM in prompt to say so when context is insufficient)
- [ ] Measure token usage per request
- [ ] Test end-to-end: upload doc → ask question → get answer

## Phase 4a — Retrieval Tuning (Manual)

- [ ] Tune chunk size & overlap, observe impact on answer quality
- [ ] Add metadata filtering to retrieval (filter by source, date, etc.)
- [ ] Implement Max Marginal Relevance (MMR) to reduce duplicate chunks

## Phase 4b — Advanced Retrieval (Manual)

- [ ] Implement basic reranking (retrieve top 10 → score with LLM → return top 3)
- [ ] Add keyword search with simple TF-IDF/BM25 (`rank-bm25` package)
- [ ] Combine vector + keyword results manually (hybrid search)
- [ ] Implement query rewriting (call LLM to rewrite vague queries before retrieval)

## Phase 5 — Conversation & UX (Manual)

- [ ] Implement chat history storage (in-memory dict per session)
- [ ] Prepend conversation history to prompt manually
- [ ] Handle follow-up questions (use history to resolve "it", "that", etc.)
- [ ] Implement streaming with OpenAI `stream=True` + FastAPI `StreamingResponse`
- [ ] Limit conversation history (sliding window / token budget)
- [ ] Add citation system (track which chunks were used, return source references)
- [ ] Better prompt engineering (few-shot examples, guardrails)

---

# Part B — LangChain / LangGraph Upgrade

## Phase 6 — Migrate to LangChain

- [ ] Replace manual document loaders with LangChain loaders
- [ ] Replace manual chunking with `RecursiveCharacterTextSplitter`
- [ ] Replace manual embeddings + Chroma with LangChain's `Chroma` vectorstore
- [ ] Replace manual RAG pipeline with LangChain's chain (retriever → prompt → LLM)
- [ ] Compare: what LangChain adds vs. what you built manually
- [ ] Integrate LangSmith for tracing & debugging chains

## Phase 7 — Evaluation & Observability

- [ ] Log all queries, retrieved chunks, and responses
- [ ] Add evaluation with RAGAS (faithfulness, relevance, context precision)
- [ ] Build a small golden dataset (20-30 Q&A pairs) for regression testing
- [ ] Add latency tracking per pipeline step
- [ ] Use LangSmith traces to debug bad answers
- [ ] Cost monitoring (LLM + embeddings usage)
- [ ] Prompt versioning

## Phase 8 — Agent Layer (LangGraph)

- [ ] Build retrieval agent with LangGraph (decides if/how to retrieve)
- [ ] Add tool-calling: `search_docs`, `get_metadata`, `summarize_doc`
- [ ] Multi-step QA agent (plan → retrieve → answer → verify)
- [ ] Retry / fallback logic when retrieval quality is low

## Phase 9 — Production Hardening

- [ ] Add PostgreSQL for metadata & chat history persistence
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
