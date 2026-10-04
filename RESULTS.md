# Evaluation results

Run date: 2026-10-04 (timestamps in the reports are UTC, 2026-10-03 23:28
to 23:36). Every number below is copied from the output of
`python -m eval.run_eval`; nothing is estimated.

The raw per-question reports behind every table are in
`eval/published_results/` (`report_*.json` for the original set,
`heldout_baseline.json` for the held-out set, `before_fix/` for the runs made
before the verifier fix).

## What was measured

- **Corpus:** 5 **synthetic** sample PDFs in `data/raw_pdfs/sample_corpus/`,
  written to resemble 3GPP specifications. They are not official 3GPP
  documents. The index holds 32 chunks.
- **Eval set:** `eval/eval_dataset.json`, 18 answerable (in-scope) questions
  and 6 adversarial (out-of-scope) questions. This is a small demonstration
  set, **not a benchmark**.
- **Models:** generation `openai/gpt-oss-20b` on Groq; embeddings
  `sentence-transformers/all-MiniLM-L6-v2`; reranker
  `cross-encoder/ms-marco-MiniLM-L-6-v2`; verifier
  `cross-encoder/nli-deberta-v3-small`.
- **Tests:** `pytest` reports 49 passed.

## Commands

```bash
python -m venv .venv
pip install -r requirements-dev.txt
pytest
python -m app.ingestion.build_index
```

`.env` needs `GROQ_API_KEY=<key>`.

```bash
python -m eval.run_eval --delay 8
ENABLE_RERANKER=false python -m eval.run_eval --delay 8
ENABLE_NLI_VERIFIER=false python -m eval.run_eval --delay 8
```

`--delay 8` paces the run for Groq's free tier (8,000 tokens per minute).
The delay happens outside the timed section, so it does not affect latency.

On Windows with Application Control enabled, `scikit-learn==1.7.2` was used
because a compiled file in 1.9.1 was blocked from loading.

## Retrieval (18 answerable questions, no LLM involved)

| Stage | Recall@5 | MRR |
| --- | --- | --- |
| FAISS only | 88.9% (16 of 18) | 0.789 |
| FAISS top-20, then cross-encoder reranker | 88.9% (16 of 18) | 0.861 |

The reranker does not change Recall@5. It moves the correct passage up for 3
questions (rank 2 to 1 twice, rank 5 to 2 once), which raises MRR. Two
questions (q15, q17) do not have the expected clause in the top 5 at either
stage.

## End to end (live LLM)

| Configuration | Keyword coverage | Wrong refusals (answerable) | Hallucination (adversarial) | Avg latency |
| --- | --- | --- | --- | --- |
| Full pipeline | **88.9%** (16 of 18) | **11.1%** (2 of 18) | **0.0%** (0 of 6) | 1.82 s |
| Reranker off | 83.3% (15 of 18) | 16.7% (3 of 18) | 0.0% (0 of 6) | 1.48 s |
| NLI verifier off | 83.3% (15 of 18) | 16.7% (3 of 18) | 0.0% (0 of 6) | 1.12 s |

- Keyword coverage: every gold keyword appears in the final answer.
- Wrong refusals: the bot declined a question it should answer.
- Hallucination: the bot answered a question the corpus cannot support.
- Latency is per question on a laptop CPU and includes local model
  inference.

The two questions the full pipeline still refuses are q2 (the verifier
scores a correct answer as unsupported) and q6 (the best passage's retrieval
score is 0.299, just under the 0.30 floor).

## Before and after the verifier fix made on this date

Same eval set, same models, full pipeline:

| | Before | After |
| --- | --- | --- |
| Keyword coverage | 50.0% (9 of 18) | 88.9% (16 of 18) |
| Wrong refusals | 44.4% (8 of 18) | 11.1% (2 of 18) |
| Hallucination | 0.0% | 0.0% |

Before the fix, all 8 wrong refusals came from the NLI verifier rejecting a
correct LLM answer, for two reasons that were fixed:

1. `app/verification/claims.py`: claims were verified with the LLM's
   formatting still attached (`**T300**`, `【1】`, `[1]`, non-breaking
   hyphens). They are now cleaned first, which also makes the final answer
   plain text.
2. `app/verification/nli_verifier.py`: each claim was compared against a
   whole multi-sentence passage or a raw markdown table. It is now compared
   against each sentence and each table row as well, and the best match
   wins.

Before the fix, the ablations read: reranker off 27.8% coverage and 66.7%
wrong refusals; NLI verifier off 66.7% coverage and 11.1% wrong refusals.

### Check that the fix did not weaken the hallucination guard

An offline check (no LLM calls) fed the verifier 15 deliberately false
answers to in-scope questions, for example "T300 is 5000 ms":

| Verifier | Correct answers accepted | False answers wrongly accepted |
| --- | --- | --- |
| Whole-passage premise only | 14 of 18 | 4 of 15 |
| Sentence and table-row premises (current) | 16 of 18 | 3 of 15 |

So the guard is not perfect: the small NLI model still accepts some false
claims that reuse the passage's own vocabulary. The 0.0% hallucination rate
above is on 6 out-of-scope questions only and does not measure this case.

## Held-out check (questions never used for tuning)

`eval/heldout_dataset.json` was written after the verifier fix: 58 new
answerable questions and 16 new adversarial questions on the same synthetic
corpus, none repeated from `eval_dataset.json`. No pipeline code or gold
answer was changed after this run.

```bash
python -m eval.run_eval --dataset eval/heldout_dataset.json --delay 6
```

| Metric (full pipeline) | Held-out set | Original set |
| --- | --- | --- |
| Keyword coverage | **93.1%** (54 of 58) | 88.9% (16 of 18) |
| Wrong refusals (answerable) | **6.9%** (4 of 58) | 11.1% (2 of 18) |
| Hallucination (adversarial) | **0.0%** (0 of 16) | 0.0% (0 of 6) |
| Recall@5, FAISS only | 87.9% (51 of 58) | 88.9% |
| MRR, FAISS only | 0.779 | 0.789 |
| MRR, after reranking | 0.853 | 0.861 |
| Avg latency | 1.87 s | 1.82 s |

- The held-out numbers match or beat the original set, so the fix was not
  overfitted to the 18 original questions.
- The four refused questions are h5, h9, h25 and h26.
- All 7 retrieval misses (h29 to h32, h44 to h46) have one cause: the PDF
  chunker files clause 6.1.3 of TS 33.501 under clause "0" and Annex A of
  TS 38.321 under clause 5.4.3. The right text is retrieved but carries the
  wrong clause label, so it would also be cited wrongly. This chunker bug is
  **not fixed**.
- **The reranker-off and verifier-off ablations were not completed on the
  held-out set.** Groq's free-tier daily quota (200,000 tokens per day for
  `openai/gpt-oss-20b`) ran out during them. Rerun them with the commands in
  the "Commands" section plus `--dataset eval/heldout_dataset.json`.
- The questions were written by the same person who made the fix, from the
  same synthetic corpus. This is a held-out set, not an independent one.

## Other changes made on this date

- `eval/run_eval.py`: added the after-reranking retrieval metric, `--delay`,
  `--dataset`, and retry on HTTP 429.
- `app/tracing.py`, `app/pipeline.py`: optional LangSmith tracing of the
  retrieve, rerank, generate and verify steps. It is off unless
  `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY` are set. **It has not been
  run against LangSmith**; only the no-op path is covered by the test suite.

## Caveats

- The verifier fix was developed by inspecting failures on the original
  18-question set, so its "after" numbers there are optimistic. The held-out
  set above is the fairer measure.
- With 18 and 6 questions, one question moves a rate by 5.6 or 16.7
  percentage points. The ablation differences (one question each) are within
  run-to-run noise: two identical baseline runs before the fix gave 50.0%
  and 44.4% wrong refusals.
