# Resources

Curated sources for feeding real content into the RAG pipeline and testing retrieval at scale.

## What to look for

Good candidates:
- Technical books / textbooks — dense, factual, citeable (100–500 pages)
- Long research papers or surveys — 30–80 page review articles
- Multi-chapter documentation — e.g. a framework's full PDF docs

Avoid:
- Scanned PDFs (image-based — `pypdf` won't extract text)
- Heavily formatted reports with tables / figures (garbled extraction)
- Novels / narrative prose (hard to write fact-based benchmark questions)

## Free sources

| Source | What's there |
|---|---|
| [arXiv.org](https://arxiv.org/) | Free research papers. Search for "survey" — surveys are long (50–100 pages) and dense. |
| [Project Gutenberg](https://www.gutenberg.org/) | Free books in text or PDF. Classic non-fiction works well. |
| [OpenStax](https://openstax.org/) | Free college textbooks (physics, economics, psychology) — excellent for QA benchmarks. |
| [NIST publications](https://www.nist.gov/publications) | Long, factual, public-domain technical reports. |
| [NASA Technical Reports Server](https://ntrs.nasa.gov/) | Engineering / science reports, public domain. |
| [WHO publications](https://www.who.int/publications) | Health / policy reports, factual. |

## Concrete suggestions that work well for RAG testing

1. **"Attention Is All You Need" + follow-up transformer papers** — 5-10 related papers make a cohesive small corpus
2. **Stanford CS224N lecture notes** (PDFs, freely available)
3. **OpenStax textbook** — e.g. *Principles of Economics* (~1100 pages), all factual, public domain
4. **The Python Language Reference** as PDF — huge, dense, factual
5. **A survey paper from arXiv** — e.g. "A Survey of Large Language Models"

## What changes in the benchmark at scale

Once you have ~5,000+ chunks indexed:
- `top_k=3` will likely feel too tight on vague questions
- Real differences between top_k=3, 5, 8, 12 become visible
- MMR starts showing benefits (diverse chunks instead of 5 paragraphs from the same section)
- Question quality matters more — write questions where the answer exists but isn't obvious

## Workflow when loading new data

1. Drop new PDFs/TXTs into `data/uploads/` and send them through `/upload`
2. Verify chunk count in Chroma (add a debug endpoint or log it during ingestion)
3. Rewrite [benchmark/questions.json](benchmark/questions.json) — current questions are specific to the Attention paper and won't transfer
4. Re-run sweep with wider range: `python benchmark/run_benchmark.py --top-k 3 5 8 12 --label wide-sweep`

## How to read LangSmith traces for cost + latency

Every `/ask` request creates one trace in LangSmith (when `LANGSMITH_TRACING=true`).
Open https://smith.langchain.com → your project → click any trace.

### Trace anatomy

```
RunnableSequence                   ← the full qa_chain (prompt | llm)
├── ChatPromptTemplate             ← prompt construction (microseconds, no cost)
└── ChatOpenAI                     ← the actual LLM call
    ├── Latency: 1.2s
    ├── Input tokens:  847
    ├── Output tokens: 193
    └── Total tokens:  1040
```

The **ChatOpenAI** node is the only one that costs money.

### Reading cost

LangSmith shows token counts per run. To calculate cost manually:
- `gpt-4o-mini`: ~$0.15 / 1M input tokens, ~$0.60 / 1M output tokens
- A typical `/ask` with top_k=5 uses ~600–900 input tokens + ~150–250 output tokens ≈ **$0.0003 per request**

To find expensive runs: sort traces by **Total Tokens** descending. Long conversation histories are the usual culprit — the sliding window (`MAX_HISTORY_MESSAGES`) keeps this bounded.

### Reading latency

Each node shows wall-clock time. The split to watch:

| Node | Expected | Red flag |
|---|---|---|
| `ChatPromptTemplate` | < 5ms | — |
| `ChatOpenAI` (streaming off) | 1–4s | > 8s |
| Full trace | 2–6s | > 10s |

High latency on `ChatOpenAI` is almost always network / OpenAI load — not your code. If the prompt template node is slow, you have a Python-side bottleneck.

### Finding retrieval quality issues

LangSmith traces the LLM call but **not** the Chroma retrieval step (that's plain Python, not a LangChain runnable). To trace retrieval:
1. Wrap `search_chunks` in a `@traceable` decorator from `langsmith`
2. Or add a log line with the retrieved chunk sources before calling the chain — visible in the trace's metadata

### Comparing runs

Use LangSmith's **Datasets & Experiments** tab to run the golden set directly through the chain and track scores over time. Alternatively, the `benchmark/results/` JSON files from `eval_retrieval.py` give you a simpler diff without the LangSmith UI.

## Notes

- Citations in LangSmith traces will start looking meaningful once filenames are real (`llm-survey.pdf` instead of `test.pdf`)
- Tokenization densities vary by language: 500 chars ≈ 100–125 tokens for English, ~250 tokens for dense code/numbers, ~400 tokens for Cyrillic/CJK
- Chunk size is fixed at ingestion time — changing it requires clearing the Chroma collection and re-uploading
