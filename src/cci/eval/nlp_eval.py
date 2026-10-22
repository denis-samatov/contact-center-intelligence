"""Intent, extraction, contradiction, preference and selective-automation evaluation.

Conditions (the same calls under different input sources):

* ``reference``  - gold transcript with gold roles. Rule extraction is near-perfect here *by
  construction* (rules and generator share the same dialogue grammar); reported as an upper
  bound, not as evidence of robustness.
* ``asr``        - GigaAM-v2 transcript of the rendered audio, roles from diarization.
* ``paraphrase`` - client turns rewritten by an LLM (facts verified to be preserved), i.e.
  phrasings neither the rules nor the classifiers were written against.
"""

from __future__ import annotations

import datetime as dt
import time

import numpy as np

from cci.domain.schema import Intent, fields_for
from cci.eval.metrics import bootstrap_ci, extraction_prf, macro_f1, set_prf
from cci.nlp.case import ReviewPolicy, build_case
from cci.nlp.rules import Extraction
from cci.nlp.rules import extract as rule_extract


def intent_scores(gold: list[str], pred: list[str]) -> dict:
    acc = [float(g == p) for g, p in zip(gold, pred, strict=True)]
    lo, hi = bootstrap_ci(acc)
    idx = list(range(len(gold)))
    mf_lo, mf_hi = bootstrap_ci(idx, stat=lambda s: macro_f1([gold[i] for i in s], [pred[i] for i in s]))
    return {
        "accuracy": sum(acc) / len(acc),
        "accuracy_ci95": [lo, hi],
        "macro_f1": macro_f1(gold, pred),
        "macro_f1_ci95": [mf_lo, mf_hi],
        "n": len(gold),
    }


def run_intent(models: dict, items: list[dict], conditions: list[str]) -> dict:
    out = {}
    gold = [it["label"]["intent"] for it in items]
    for name, model in models.items():
        for cond in conditions:
            texts = [it[f"text_{cond}"] for it in items]
            t0 = time.perf_counter()
            pred, probs = model.predict_proba(texts)
            out[f"{name}/{cond}"] = {
                **intent_scores(gold, pred),
                "seconds_per_call": (time.perf_counter() - t0) / len(items),
            }
    return out


def _filter(ex: Extraction, intent: str) -> dict:
    allowed = set(fields_for(Intent(intent)))
    return {k: v for k, v in ex.fields.items() if k in allowed}


def run_extraction(items: list[dict], conditions: list[str], extractors: dict) -> dict:
    """Field scoring uses the *gold* intent's schema to isolate extraction quality."""
    out = {}
    for name, fn in extractors.items():
        for cond in conditions:
            G, P, X, CG, CP, PG, PP, secs = [], [], [], [], [], [], [], 0.0
            for it in items:
                lab = it["label"]
                t0 = time.perf_counter()
                ex = fn(it[f"turns_{cond}"], dt.date.fromisoformat(it["call_date"]))
                secs += time.perf_counter() - t0
                contra = {c["field"] for c in lab["contradictions"]}
                G.append(lab["fields"])
                P.append(_filter(ex, lab["intent"]))
                X.append(contra)
                CG.append(contra)
                CP.append({c.field for c in ex.contradictions if c.field in fields_for(Intent(lab["intent"]))})
                PG.append({f"{k}={v}" for k, v in lab["preferences"].items() if v})
                PP.append({f"{k}={v}" for k, v in ex.preferences.model_dump().items() if v})
            out[f"{name}/{cond}"] = {
                "fields": extraction_prf(G, P, X),
                "contradictions": set_prf(CG, CP),
                "preferences": set_prf(PG, PP),
                "seconds_per_call": secs / len(items),
            }
    return out


def case_error(case, lab: dict) -> bool:
    """An automated case is wrong if the intent, the queue or any required field is wrong."""
    if case.intent != lab["intent"] or case.queue != lab["queue"]:
        return True
    contra = {c["field"] for c in lab["contradictions"]}
    from cci.domain.schema import REQUIRED
    from cci.eval.metrics import field_match

    for f in REQUIRED[Intent(lab["intent"])]:
        if f in contra:
            continue
        g = lab["fields"].get(f)
        if g is None:
            continue
        if not field_match(f, g, case.fields.get(f)):
            return True
    return False


def run_selective(intent_model, dev: list[dict], test: list[dict], cond: str, target_error: float = 0.02) -> dict:
    """Choose the intent-probability threshold on dev, report automation/error on test.

    Compared policies: (a) automate everything; (b) review only on missing/contradiction;
    (c) full selective policy (b + confidence threshold from dev)."""

    def decide(items, policy):
        labels, probs = intent_model.predict_proba([it[f"text_{cond}"] for it in items])
        rows = []
        for it, lab, p in zip(items, labels, probs, strict=True):
            ex = rule_extract(it[f"turns_{cond}"], dt.date.fromisoformat(it["call_date"]))
            case = build_case(lab, float(np.max(p)), ex, policy)
            rows.append((case, case_error(case, it["label"])))
        return rows

    def summary(rows):
        auto = [e for c, e in rows if c.decision == "auto"]
        return {
            "automation_rate": len(auto) / len(rows),
            "error_rate_on_automated": (sum(auto) / len(auto)) if auto else None,
            "errors_automated": sum(auto),
            "automated": len(auto),
            "n": len(rows),
            "overall_case_error_if_all_automated": sum(e for _, e in rows) / len(rows),
        }

    # threshold selection on dev
    dev_rows = decide(dev, ReviewPolicy(min_intent_prob=0.0))
    best_t, curve = 1.0, []
    for t in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        sel = [
            (c, e)
            for c, e in dev_rows
            if c.intent_prob >= t and not (set(c.review_reasons) - {"low_intent_confidence"})
        ]
        err = sum(e for _, e in sel) / len(sel) if sel else 0.0
        curve.append({"threshold": float(t), "dev_automation": len(sel) / len(dev_rows), "dev_error": err})
        if sel and err <= target_error and t < best_t:
            best_t = float(t)
    if not any(p["dev_error"] <= target_error and p["dev_automation"] > 0 for p in curve):
        best_t = 1.01
    return {
        "condition": cond,
        "target_error_on_dev": target_error,
        "chosen_threshold": best_t,
        "dev_curve": curve,
        "test_automate_all": summary(
            [
                (type(c)(**{**c.__dict__, "decision": "auto"}), e)
                for c, e in decide(test, ReviewPolicy(min_intent_prob=0.0))
            ]
        ),
        "test_rules_only": summary(decide(test, ReviewPolicy(min_intent_prob=0.0))),
        "test_selective": summary(decide(test, ReviewPolicy(min_intent_prob=best_t))),
    }
