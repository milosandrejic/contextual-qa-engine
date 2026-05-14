"""
Retrieval evaluation runner.

Loads the golden set, runs each question through the retriever and optionally the
full RAG pipeline, computes retrieval + RAGAS metrics, and writes a timestamped
JSON report to benchmark/results/.

Usage (from project root with .venv active):

    # Retrieval-only (fast, no LLM calls beyond embeddings)
    python scripts/eval_retrieval.py

    # Full end-to-end including RAGAS (slower, costs ~$0.01 per run)
    python scripts/eval_retrieval.py --ragas

    # Custom top-k
    python scripts/eval_retrieval.py --top-k 5 --ragas

    # Custom golden set path
    python scripts/eval_retrieval.py --golden-set benchmark/golden_set.json
"""

import argparse
import asyncio
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

# Allow importing from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.services.pg_vector_store import get_retriever
from app.services.prompt import build_context
from app.services.llm import generate_answer
from benchmark.metrics import compute_retrieval_metrics, aggregate_metrics

RESULTS_DIR = Path("benchmark/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_golden_set(path: str) -> list[dict]:
    with open(path) as f:
        return json.load(f)


async def run_retrieval_eval(
    golden_set: list[dict],
    top_k: int,
    include_ragas: bool,
    mode: str = "hybrid",
) -> dict:
    per_query_retrieval: list[dict] = []
    ragas_samples: list[dict] = []

    retrieve = get_retriever(mode)
    print(f"Running evaluation on {len(golden_set)} questions (top_k={top_k}, mode={mode})...")

    for i, item in enumerate(golden_set, start=1):
        question = item["question"]
        expected = item["expected_source_chunks"]

        print(f"  [{i}/{len(golden_set)}] {question[:70]}...")

        retrieved = await retrieve(query=question, top_k=top_k)
        metrics = compute_retrieval_metrics(retrieved=retrieved, expected=expected)
        per_query_retrieval.append({"id": item["id"], **metrics})

        if include_ragas:
            context = build_context(retrieved)
            result = generate_answer(context=context, question=question)
            ragas_samples.append({
                "question": question,
                "answer": result["answer"],
                "contexts": [c["text"] for c in retrieved],
            })

    metrics_only = [{k: v for k, v in m.items() if k != "id"} for m in per_query_retrieval]
    retrieval_summary = aggregate_metrics(metrics_only)

    report = {
        "run_at": datetime.now(timezone.utc).isoformat(),
        "retrieval_mode": mode,
        "top_k": top_k,
        "num_questions": len(golden_set),
        "retrieval_metrics": retrieval_summary,
        "per_query": per_query_retrieval,
        "ragas_metrics": None,
    }

    if include_ragas and ragas_samples:
        print("\nRunning RAGAS end-to-end evaluation...")
        from benchmark.ragas_metrics import evaluate_with_ragas
        ragas_scores = evaluate_with_ragas(ragas_samples)
        report["ragas_metrics"] = ragas_scores

    return report


def save_report(report: dict) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = RESULTS_DIR / f"eval_{timestamp}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    return out_path


def print_summary(report: dict) -> None:
    print("\n" + "=" * 50)
    print("RETRIEVAL METRICS")
    print("=" * 50)
    for key, value in report["retrieval_metrics"].items():
        print(f"  {key:<15} {value:.4f}")

    if report["ragas_metrics"]:
        print("\nRAGAS METRICS")
        print("=" * 50)
        for key, value in report["ragas_metrics"].items():
            print(f"  {key:<25} {value:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run retrieval evaluation against golden set.")
    parser.add_argument(
        "--golden-set",
        default="benchmark/golden_set.json",
        help="Path to golden set JSON file (default: benchmark/golden_set.json)",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of chunks to retrieve per question (default: 5)",
    )
    parser.add_argument(
        "--ragas",
        action="store_true",
        help="Also run RAGAS end-to-end evaluation (requires LLM calls)",
    )
    parser.add_argument(
        "--mode",
        default="hybrid",
        choices=["semantic", "lexical", "hybrid"],
        help="Retrieval mode to benchmark (default: hybrid)",
    )
    args = parser.parse_args()

    golden_set = load_golden_set(args.golden_set)

    report = asyncio.run(run_retrieval_eval(
        golden_set=golden_set,
        top_k=args.top_k,
        include_ragas=args.ragas,
        mode=args.mode,
    ))

    out_path = save_report(report)
    print_summary(report)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
