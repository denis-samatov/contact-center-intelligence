"""LLM paraphrases of client turns, used as a distribution-shift test set.

Each client turn is rewritten by the LLM. A paraphrase is accepted only if it preserves the
facts: the same multiset of number expressions (after parsing) and every gazetteer entity
(city, road, street) and key cue word of the original. Rejected paraphrases keep the
original turn. Operator turns are not changed (scripted in real contact centers too).
Acceptance statistics are part of the result.
"""

from __future__ import annotations

import json
import re
from collections import Counter

from pydantic import BaseModel

from cci.domain.ru_numbers import parse_numbers
from cci.llm.client import LLMError
from cci.synth import templates as T

KEEP_CUES = [
    "не помню",
    "не знаю",
    "не могу сказать",
    "извините",
    "ошибся",
    "эсэмэс",
    "почт",
    "перезвоните",
    "утром",
    "днем",
    "вечером",
    "травмпункт",
    "пострадав",
    "второй автомобиль",
    "один был",
]


class Para(BaseModel):
    text: str


def _facts(text: str) -> tuple[Counter, set[str]]:
    t = text.lower().replace("ё", "е")
    nums = Counter(n.text for n in parse_numbers(t))
    ents = {e.lower() for e in T.CITIES + T.ROADS + T.STREETS if e.lower() in t}
    cues = {c for c in KEEP_CUES if c in t}
    return nums, ents | cues


INSTRUCTION = (
    "Перескажи реплику клиента другими словами, разговорным русским языком, сохранив смысл. "
    "Обязательно измени формулировку. Все числа, даты и номера оставь в точности как есть, словами. "
    "Названия городов и улиц не меняй. Не добавляй новых фактов. Ответь JSON с полем text."
)
# Few-shot pairs as separate chat turns: inlined examples were copied verbatim by the 1.7B model.
FEW_SHOT = [
    (
        "Здравствуйте, хочу продлить полис, срок заканчивается.",
        "Добрый день, у меня скоро кончается страховка, хотел бы ее продлить.",
    ),
    (
        "Номер полиса ноль два три пять семь один один восемь.",
        "Да, сейчас скажу номер: ноль два три пять семь один один восемь.",
    ),
]


def _prompt(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": "Ты переписываешь реплики клиентов страховой компании другими словами."}]
    for src, dst in FEW_SHOT:
        msgs.append({"role": "user", "content": f"{INSTRUCTION}\nРеплика: {src}"})
        msgs.append({"role": "assistant", "content": json.dumps({"text": dst}, ensure_ascii=False)})
    msgs.append({"role": "user", "content": f"{INSTRUCTION}\nРеплика: {text}"})
    return msgs


def _norm(s: str) -> str:
    return re.sub(r"[^\w ]", "", s.lower().replace("ё", "е")).strip()


_EXAMPLE_OUTPUTS = {_norm(d) for _, d in FEW_SHOT}


def paraphrase_calls(calls: list[dict], llm, embed=None, min_similarity: float = 0.0) -> dict:
    """Paraphrase client turns; ``embed`` (texts -> vectors) enables a semantic-similarity gate."""
    out, accepted, rejected, failed = [], 0, 0, 0
    reasons = Counter()
    samples: list[dict] = []
    sims: list[float] = []
    for c in calls:
        turns = []
        for t in c["turns"]:
            if t["speaker"] != "client" or len(t["text"].split()) < 3:
                turns.append([t["speaker"], t["text"]])
                continue
            try:
                para = llm.structured(_prompt(t["text"]), Para, max_tokens=160, temperature=0.7).text.strip()
            except LLMError:
                failed += 1
                turns.append([t["speaker"], t["text"]])
                continue
            n0, e0 = _facts(t["text"])
            n1, e1 = _facts(para)
            copied = any(e in _norm(para) for e in _EXAMPLE_OUTPUTS)
            sim = _cosine(embed, t["text"], para) if embed else 1.0
            sims.append(sim)
            ok = (
                n0 == n1
                and e0 <= e1
                and not copied
                and sim >= min_similarity
                and _norm(para) != _norm(t["text"])
                and len(para) < 3 * len(t["text"]) + 40
            )
            if ok:
                accepted += 1
                turns.append([t["speaker"], re.sub(r"\s+", " ", para)])
            else:
                rejected += 1
                if copied:
                    why = "copied_example"
                elif n0 != n1:
                    why = "numbers_changed"
                elif not e0 <= e1:
                    why = "entity_or_cue_lost"
                elif _norm(para) == _norm(t["text"]):
                    why = "unchanged"
                elif sim < min_similarity:
                    why = "low_similarity"
                else:
                    why = "too_long"
                reasons[why] += 1
                if len(samples) < 12:
                    samples.append({"original": t["text"], "paraphrase": para, "reason": why, "similarity": sim})
                turns.append([t["speaker"], t["text"]])
        out.append({"call_id": c["call_id"], "turns": turns})
    total = accepted + rejected + failed
    return {
        "calls": out,
        "client_turns": total,
        "accepted": accepted,
        "rejected": rejected,
        "llm_failures": failed,
        "acceptance_rate": accepted / total if total else 0.0,
        "rejection_reasons": dict(reasons),
        "rejected_samples": samples,
        "min_similarity": min_similarity,
        "similarity_quantiles": _quantiles(sims),
        "llm_usage": {
            "calls": llm.usage.calls,
            "prompt_tokens": llm.usage.prompt_tokens,
            "completion_tokens": llm.usage.completion_tokens,
            "seconds": llm.usage.seconds,
        },
    }


def _cosine(embed, a: str, b: str) -> float:
    va, vb = embed([a, b])
    num = sum(x * y for x, y in zip(va, vb, strict=True))
    den = (sum(x * x for x in va) ** 0.5) * (sum(y * y for y in vb) ** 0.5)
    return num / den if den else 0.0


def _quantiles(xs: list[float]) -> dict:
    if not xs:
        return {}
    s = sorted(xs)
    return {f"p{q}": round(s[min(len(s) - 1, int(q / 100 * len(s)))], 4) for q in (5, 25, 50, 75)}
