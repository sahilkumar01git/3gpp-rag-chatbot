"""
Evaluation harness.

    python -m eval.run_eval                  # full run: retrieval metrics + end-to-end
                                               # generation/hallucination metrics (needs
                                               # GROQ_API_KEY + real embedding/NLI models)
    python -m eval.run_eval --skip-generation # retrieval metrics only, no LLM calls
    python -m eval.run_eval --offline         # harness self-test with a deterministic
                                               # fake embedder, no network/API needed at all

Two independent measurements are produced:

1. RETRIEVAL METRICS (Recall@k, MRR) — for every in-scope question in
   eval/eval_dataset.json, checks whether the passage carrying the
   expected clause is retrieved in the top-k FAISS results at all
   (Recall@k) and how highly it ranks (Mean Reciprocal Rank). This
   measures the retrieval+reranking stage in isolation.

2. END-TO-END METRICS (needs a configured Groq key and real models,
   skipped otherwise):
     - keyword_coverage: for in-scope questions, whether the gold
       keywords appear in the final (post-verification) answer.
     - answerable_refusal_rate: how often the bot refuses to answer an
       in-scope question it should be able to answer (ideally near 0).
     - hallucination_rate: the complement of the refusal rate on the
       OUT-OF-SCOPE / adversarial question set — i.e. how often the bot
       fabricates an answer to a question it has no real evidence for
       (ideally 0). This is the headline number for "is the hallucination
       guard actually working".

Results are written to eval/results/report.json and report.md.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from app.config import settings
from app.logging_config import configure_logging
from app.pipeline import RagPipeline
from app.retrieval.embedder import SentenceTransformerEmbedder

logger = logging.getLogger(__name__)

EVAL_DIR = Path(__file__).resolve().parent
DATASET_PATH = EVAL_DIR / "eval_dataset.json"
RESULTS_DIR = EVAL_DIR / "results"


def load_dataset() -> dict:
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def _clause_matches(retrieved_clause: str, expected_clause: str) -> bool:
    # A long clause may have been split into part_index sub-chunks, and
    # table chunks are named "Table@pN" — exact match on the clause number
    # string is what we want; sub-splitting doesn't change clause_number.
    return retrieved_clause == expected_clause


def evaluate_retrieval(pipeline: RagPipeline, dataset: dict, top_k: int) -> dict:
    details = []
    hits = 0
    reciprocal_ranks = []

    for item in dataset["in_scope"]:
        query_vector = pipeline.embedder.encode([item["question"]])[0]
        results = pipeline.vector_store.search(query_vector, top_n=top_k)

        rank = None
        for i, r in enumerate(results, start=1):
            if r.metadata.spec_id == item["expected_spec_id"] and _clause_matches(r.metadata.clause_number, item["expected_clause"]):
                rank = i
                break

        if rank is not None:
            hits += 1
            reciprocal_ranks.append(1.0 / rank)
        else:
            reciprocal_ranks.append(0.0)

        details.append(
            {
                "id": item["id"],
                "question": item["question"],
                "expected": f"{item['expected_spec_id']} Clause {item['expected_clause']}",
                "found_at_rank": rank,
                "top_result": f"{results[0].metadata.spec_id} Clause {results[0].metadata.clause_number}" if results else None,
            }
        )

    n = len(dataset["in_scope"])
    return {
        "top_k": top_k,
        "recall_at_k": round(hits / n, 4) if n else 0.0,
        "mrr": round(sum(reciprocal_ranks) / n, 4) if n else 0.0,
        "n_questions": n,
        "details": details,
    }


def evaluate_end_to_end(pipeline: RagPipeline, dataset: dict) -> dict:
    in_scope_details = []
    keyword_hits = 0
    answerable_refusals = 0
    latencies = []

    for item in dataset["in_scope"]:
        result = pipeline.answer(item["question"])
        latencies.append(result.latency_seconds)
        answer_lower = result.answer.lower()
        keywords_found = [kw for kw in item["gold_answer_keywords"] if kw.lower() in answer_lower]
        all_found = len(keywords_found) == len(item["gold_answer_keywords"])
        if all_found:
            keyword_hits += 1
        if result.refused:
            answerable_refusals += 1

        in_scope_details.append(
            {
                "id": item["id"],
                "question": item["question"],
                "refused": result.refused,
                "confidence": result.overall_confidence,
                "keywords_expected": item["gold_answer_keywords"],
                "keywords_found": keywords_found,
                "fully_supported": all_found,
                "answer": result.answer,
            }
        )

    adversarial_details = []
    hallucinations = 0
    for item in dataset["out_of_scope"]:
        result = pipeline.answer(item["question"])
        latencies.append(result.latency_seconds)
        if not result.refused:
            hallucinations += 1
        adversarial_details.append(
            {
                "id": item["id"],
                "question": item["question"],
                "note": item["note"],
                "refused": result.refused,
                "confidence": result.overall_confidence,
                "answer": result.answer,
            }
        )

    n_in_scope = len(dataset["in_scope"])
    n_adversarial = len(dataset["out_of_scope"])

    return {
        "keyword_coverage_rate": round(keyword_hits / n_in_scope, 4) if n_in_scope else 0.0,
        "answerable_refusal_rate": round(answerable_refusals / n_in_scope, 4) if n_in_scope else 0.0,
        "hallucination_rate": round(hallucinations / n_adversarial, 4) if n_adversarial else 0.0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 3) if latencies else 0.0,
        "in_scope_details": in_scope_details,
        "adversarial_details": adversarial_details,
    }


def write_report(report: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        f"# Evaluation report — {report['timestamp']}",
        "",
        f"Mode: **{report['mode']}**",
        "",
        "## Retrieval metrics",
        "",
        f"- Recall@{report['retrieval']['top_k']}: **{report['retrieval']['recall_at_k']:.1%}**",
        f"- Mean Reciprocal Rank: **{report['retrieval']['mrr']:.3f}**",
        f"- Questions evaluated: {report['retrieval']['n_questions']}",
        "",
    ]

    if report.get("end_to_end"):
        e2e = report["end_to_end"]
        lines += [
            "## End-to-end metrics",
            "",
            f"- Keyword coverage on in-scope questions: **{e2e['keyword_coverage_rate']:.1%}**",
            f"- Refusal rate on answerable (in-scope) questions: **{e2e['answerable_refusal_rate']:.1%}** (lower is better)",
            f"- Hallucination rate on out-of-scope questions: **{e2e['hallucination_rate']:.1%}** (lower is better — 0% means every "
            "out-of-scope question was correctly refused)",
            f"- Average latency: {e2e['avg_latency_seconds']:.2f}s",
            "",
        ]
    else:
        lines += [
            "## End-to-end metrics",
            "",
            "_Skipped — no Groq API key configured, `--skip-generation` was passed, or `--offline` mode was used._",
            "",
        ]

    (RESULTS_DIR / "report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Evaluate the 3GPP RAG chatbot.")
    parser.add_argument("--top-k", type=int, default=5, help="k for Recall@k / retrieval evaluation.")
    parser.add_argument("--skip-generation", action="store_true", help="Only run retrieval metrics (no LLM calls).")
    parser.add_argument("--offline", action="store_true", help="Use a deterministic fake embedder; skip generation entirely.")
    args = parser.parse_args()

    if args.offline:
        import sys

        sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
        from tests.fakes import FakeEmbedder

        embedder = FakeEmbedder()
        print("Running in --offline mode: retrieval metrics use a deterministic fake embedder "
              "(NOT representative of real retrieval quality — this only self-tests the harness). "
              "End-to-end generation is always skipped in this mode.\n")
    else:
        embedder = SentenceTransformerEmbedder.get_singleton(settings.embed_model_name)

    pipeline = RagPipeline(embedder=embedder)
    dataset = load_dataset()

    print(f"Loaded {len(dataset['in_scope'])} in-scope and {len(dataset['out_of_scope'])} out-of-scope eval questions.")
    print("Running retrieval evaluation ...")
    retrieval_metrics = evaluate_retrieval(pipeline, dataset, top_k=args.top_k)
    print(f"  Recall@{args.top_k}: {retrieval_metrics['recall_at_k']:.1%}   MRR: {retrieval_metrics['mrr']:.3f}")

    end_to_end_metrics = None
    run_generation = not args.offline and not args.skip_generation
    if run_generation and not settings.groq_api_key:
        print("\nGROQ_API_KEY not set — skipping end-to-end generation/hallucination evaluation. "
              "Set it in .env to run the full evaluation.")
        run_generation = False

    if run_generation:
        print("\nRunning end-to-end generation + hallucination evaluation (this calls the Groq API for every question) ...")
        end_to_end_metrics = evaluate_end_to_end(pipeline, dataset)
        print(f"  Keyword coverage: {end_to_end_metrics['keyword_coverage_rate']:.1%}")
        print(f"  Answerable-question refusal rate: {end_to_end_metrics['answerable_refusal_rate']:.1%}")
        print(f"  Hallucination rate (out-of-scope): {end_to_end_metrics['hallucination_rate']:.1%}")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "mode": "offline (fake embedder, harness self-test)" if args.offline else "live",
        "retrieval": retrieval_metrics,
        "end_to_end": end_to_end_metrics,
    }
    write_report(report)
    print(f"\nFull report written to {RESULTS_DIR / 'report.json'} and {RESULTS_DIR / 'report.md'}")


if __name__ == "__main__":
    main()
