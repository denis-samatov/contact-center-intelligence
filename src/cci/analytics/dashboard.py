"""Quality and performance dashboard data.

Two clearly separated parts:

* **operations** - measured on the calls processed by *this* deployment (database): volume by
  status, processing time and real-time factor, automation and review reasons, queues,
  repeat contacts, operator load.
* **offline quality** - headline numbers of the committed evaluation in ``results/`` (synthetic
  test split), each with the file it comes from. They describe the models, not this traffic.
"""

from __future__ import annotations

import json
from pathlib import Path


def _pct(xs: list[float], q: float) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(q * len(s)))]


def operations(calls: list[dict], report: dict) -> dict:
    """``calls``: [{status, duration_s, timings}], ``report``: :func:`cci.analytics.metrics.daily_report`."""
    by_status: dict[str, int] = {}
    proc, rtf = [], []
    for c in calls:
        by_status[c["status"]] = by_status.get(c["status"], 0) + 1
        if c["status"] == "done" and c.get("timings"):
            t = sum(c["timings"].values())
            proc.append(t)
            if c.get("duration_s"):
                rtf.append(t / c["duration_s"])
    return {
        "calls_by_status": by_status,
        "processing_s": {"p50": _pct(proc, 0.5), "p95": _pct(proc, 0.95), "n": len(proc)},
        "real_time_factor": {"p50": _pct(rtf, 0.5), "p95": _pct(rtf, 0.95)},
        **report,
    }


def _load(results: Path, name: str) -> dict | None:
    p = results / f"{name}.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def offline_quality(results: Path) -> list[dict]:
    """Rows of (area, metric, value, source). Missing files are skipped, not invented."""
    rows: list[dict] = []

    def add(area, metric, value, source):
        if value is not None:
            rows.append({"area": area, "metric": metric, "value": value, "source": f"results/{source}.json"})

    asr = _load(results, "asr_test_telephony")
    if asr:
        for b, m in asr["backends"].items():
            add("ASR (telephony, 60 calls)", f"WER {b}", m["wer"], "asr_test_telephony")
        d = asr.get("diarization") or {}
        add("Diarization", "DER", d.get("der_mean"), "asr_test_telephony")
    roles = _load(results, "roles_test")
    if roles:
        add("Diarization", "operator/client role accuracy", roles["role_model"]["mean"], "roles_test")
    nlp = _load(results, "nlp")
    if nlp:
        for k, label in (("tfidf_lr/asr:opening", "TF-IDF"), ("embedding_lr/asr:opening", "embeddings + LR")):
            if k in nlp["intent"]:
                add("Intent, first utterance (ASR)", f"macro-F1 {label}", nlp["intent"][k]["macro_f1"], "nlp")
    post = _load(results, "rules_v1_1")
    if post:
        f = post["test"]["rules/asr"]["fields"]
        add("Extraction (ASR, rules v1.1, post-hoc)", "field precision", f["precision"], "rules_v1_1")
        add("Extraction (ASR, rules v1.1, post-hoc)", "field recall", f["recall"], "rules_v1_1")
        sel = post["selective"]["test_rules_only"]
        add("Automation (review on missing/contradiction)", "automation rate", sel["automation_rate"], "rules_v1_1")
        add(
            "Automation (review on missing/contradiction)",
            "error rate on automated",
            sel["error_rate_on_automated"],
            "rules_v1_1",
        )
    rag = _load(results, "rag")
    if rag:
        add("KB answers (40 questions)", "fully correct, grounded", rag.get("grounded_fully_correct"), "rag")
        add("KB answers (40 questions)", "fully correct, closed book", rag.get("closed_book_fully_correct"), "rag")
        add("KB answers (40 questions)", "citation precision", rag.get("citation_precision"), "rag")
    return rows
