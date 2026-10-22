"""Retrieval and answer evaluation on the fictional KB.

* Retrieval: recall@k of the gold chunk for BM25, dense and hybrid.
* Answers: closed-book LLM (no retrieval) vs grounded LLM (hybrid top-3). Fact recall is the
  share of gold fact patterns found in the answer. Grounded answers are also checked by
  :func:`cci.rag.answer.verify` (citation validity, numeric support), and citation precision
  measures whether a cited chunk is a gold chunk.

The KB is fictional, so a closed-book model *cannot* know these rules: its correct answers
measure lucky guesses on generic facts (documents such as a passport), and its specific
numbers measure hallucination.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from cci.rag.answer import answer_closed_book, answer_grounded
from cci.rag.retrieval import HybridIndex, load_kb


def _fact_recall(answer: str | None, facts: list[str]) -> float:
    if not answer:
        return 0.0
    a = answer.lower().replace("ё", "е")
    return sum(bool(re.search(f, a)) for f in facts) / len(facts)


def eval_rag(kb_dir: Path, qa_path: Path, llm=None, embed=None, ks=(1, 3, 5)) -> dict:
    chunks = load_kb(kb_dir)
    t0 = time.perf_counter()
    index = HybridIndex(chunks, embed)
    index_s = time.perf_counter() - t0
    qa = [json.loads(line) for line in qa_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    modes = ["bm25"] + (["dense", "hybrid"] if embed else [])
    retrieval = {}
    for mode in modes:
        for k in ks:
            hits = sum(
                any(c.chunk_id in q["gold_chunks"] for c in index.search(q["question"], k=k, mode=mode)) for q in qa
            )
            retrieval[f"{mode}@{k}"] = hits / len(qa)
    out = {"chunks": len(chunks), "questions": len(qa), "index_build_s": index_s, "retrieval_recall": retrieval}
    if llm is None:
        return out
    mode = "hybrid" if embed else "bm25"
    rows = []
    for q in qa:
        cb = answer_closed_book(llm, q["question"])
        gr = answer_grounded(llm, index, q["question"], k=3, mode=mode)
        cited = gr.get("citations") or []
        rows.append(
            {
                "id": q["id"],
                "question": q["question"],
                "closed_book": cb.get("answer"),
                "grounded": gr.get("answer"),
                "citations": cited,
                "retrieved": gr.get("retrieved"),
                "closed_fact_recall": _fact_recall(cb.get("answer"), q["facts"]),
                "grounded_fact_recall": _fact_recall(gr.get("answer"), q["facts"]),
                "verified_grounded": gr.get("grounded", False),
                "valid_citations": gr.get("valid_citations", False),
                "unsupported_numbers": gr.get("unsupported_numbers", []),
                "citation_precision": (sum(c in q["gold_chunks"] for c in cited) / len(cited)) if cited else 0.0,
                "gold_retrieved": any(c in q["gold_chunks"] for c in gr.get("retrieved", [])),
            }
        )
    n = len(rows)
    out.update(
        {
            "answer_retrieval_mode": mode,
            "closed_book_fact_recall": sum(r["closed_fact_recall"] for r in rows) / n,
            "grounded_fact_recall": sum(r["grounded_fact_recall"] for r in rows) / n,
            "grounded_fully_correct": sum(r["grounded_fact_recall"] == 1.0 for r in rows) / n,
            "closed_book_fully_correct": sum(r["closed_fact_recall"] == 1.0 for r in rows) / n,
            "verified_grounded_rate": sum(r["verified_grounded"] for r in rows) / n,
            "citation_precision": sum(r["citation_precision"] for r in rows) / n,
            "fully_correct_when_verified": (
                sum(r["grounded_fact_recall"] == 1.0 for r in rows if r["verified_grounded"])
                / max(1, sum(r["verified_grounded"] for r in rows))
            ),
            "llm_usage": {
                "calls": llm.usage.calls,
                "prompt_tokens": llm.usage.prompt_tokens,
                "completion_tokens": llm.usage.completion_tokens,
                "seconds": llm.usage.seconds,
                "p50_latency_s": sorted(llm.usage.latencies)[len(llm.usage.latencies) // 2]
                if llm.usage.latencies
                else None,
            },
            "rows": rows,
        }
    )
    return out
