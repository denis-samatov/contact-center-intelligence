"""Evaluation metrics shared by all experiments.

Field scoring rules (fixed before the final test run):

* structured fields (numbers, dates, enums, places): exact match after normalization;
* ``new_value``: case/whitespace-insensitive match;
* ``cancellation_reason`` (free text): gold contained in prediction after normalization;
* fields involved in a gold contradiction are excluded from value scoring (there is no
  single correct value) and scored by contradiction detection instead.
"""

from __future__ import annotations

import random
import re
from collections import Counter

from cci.domain.ru_numbers import normalize_numbers

FREE_TEXT = {"cancellation_reason"}
CASELESS = {"new_value"}


def norm_text(s: str) -> str:
    s = s.lower().replace("ё", "е")
    s = re.sub(r"[^\w\s-]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def field_match(field: str, gold: str, pred: str | None) -> bool:
    if pred is None:
        return False
    if field in FREE_TEXT:
        return norm_text(gold) in norm_text(pred)
    if field in CASELESS:
        return norm_text(gold) == norm_text(pred)
    return gold == pred


def extraction_prf(golds: list[dict], preds: list[dict], contra_fields: list[set[str]]) -> dict:
    """Micro P/R/F1 over (call, field) pairs plus per-field recall."""
    tp = fp = fn = 0
    per_tp, per_n = Counter(), Counter()
    for g, p, excl in zip(golds, preds, contra_fields, strict=True):
        for f, v in g.items():
            if f in excl:
                continue
            per_n[f] += 1
            if field_match(f, v, p.get(f)):
                tp += 1
                per_tp[f] += 1
            else:
                fn += 1
                if p.get(f) is not None:
                    fp += 1
        for f in p:
            if f not in g and f not in excl and p[f] is not None:
                fp += 1
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "per_field_recall": {f: per_tp[f] / per_n[f] for f in sorted(per_n)},
        "pairs": sum(per_n.values()),
    }


def set_prf(gold: list[set], pred: list[set]) -> dict:
    tp = sum(len(g & p) for g, p in zip(gold, pred, strict=True))
    fp = sum(len(p - g) for g, p in zip(gold, pred, strict=True))
    fn = sum(len(g - p) for g, p in zip(gold, pred, strict=True))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": prec,
        "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "support": tp + fn,
    }


def macro_f1(gold: list[str], pred: list[str]) -> float:
    labels = sorted(set(gold))
    f1s = []
    for lab in labels:
        tp = sum(g == lab and p == lab for g, p in zip(gold, pred, strict=True))
        fp = sum(g != lab and p == lab for g, p in zip(gold, pred, strict=True))
        fn = sum(g == lab and p != lab for g, p in zip(gold, pred, strict=True))
        pr = tp / (tp + fp) if tp + fp else 0.0
        rc = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * pr * rc / (pr + rc) if pr + rc else 0.0)
    return sum(f1s) / len(f1s)


def bootstrap_ci(values: list[float], n: int = 2000, seed: int = 0, stat=None) -> tuple[float, float]:
    """95% percentile bootstrap CI of the mean (or of ``stat`` over resampled indices)."""
    rng = random.Random(seed)
    k = len(values)
    stats = []
    for _ in range(n):
        idx = [rng.randrange(k) for _ in range(k)]
        stats.append(stat(idx) if stat else sum(values[i] for i in idx) / k)
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


def cluster_bootstrap_ci(groups: list[str], stat, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    """95% percentile bootstrap that resamples whole groups (e.g. all calls sharing one opening
    template). Calls in a group are not independent; resampling calls would understate the
    uncertainty when there are few distinct groups."""
    rng = random.Random(seed)
    members: dict[str, list[int]] = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    keys = sorted(members)
    stats = []
    for _ in range(n):
        idx = [i for _ in keys for i in members[keys[rng.randrange(len(keys))]]]
        stats.append(stat(idx))
    stats.sort()
    return stats[int(0.025 * n)], stats[int(0.975 * n) - 1]


# --------------------------------------------------------------------------- ASR
def asr_normalize(text: str) -> str:
    """Lowercase, ё->е, digits for number words (both sides), no punctuation.

    Whisper often writes digits while transducers write words; normalizing numbers on both
    reference and hypothesis avoids penalizing the format rather than the recognition."""
    return norm_text(normalize_numbers(norm_text(text)))


def _edit_distance(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def wer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = ref.split(), hyp.split()
    return _edit_distance(r, h), len(r)


def cer_counts(ref: str, hyp: str) -> tuple[int, int]:
    r, h = list(ref.replace(" ", "")), list(hyp.replace(" ", ""))
    return _edit_distance(r, h), len(r)
