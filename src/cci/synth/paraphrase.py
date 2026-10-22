"""LLM paraphrases of client turns, used as a distribution-shift test set.

Each client turn is rewritten by the LLM. A paraphrase is accepted only if it preserves the
facts: the same multiset of number expressions (after parsing) and every gazetteer entity
(city, road, street) and key cue word of the original. Rejected paraphrases keep the
original turn. Operator turns are not changed (scripted in real contact centers too).
Acceptance statistics are part of the result.
"""

from __future__ import annotations

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


def _prompt(text: str) -> list[dict]:
    return [
        {"role": "system", "content": "Ты переписываешь реплики клиентов страховой компании другими словами."},
        {
            "role": "user",
            "content": (
                "Перефразируй реплику клиента разговорным русским языком, сохранив смысл. Все числа, даты и номера "
                "оставь в точности как есть, словами. Названия городов и улиц не меняй. Не добавляй новых фактов.\n"
                f"Реплика: {text}\nОтветь JSON с полем text."
            ),
        },
    ]


def paraphrase_calls(calls: list[dict], llm) -> dict:
    out, accepted, rejected, failed = [], 0, 0, 0
    reasons = Counter()
    for c in calls:
        turns = []
        for t in c["turns"]:
            if t["speaker"] != "client" or len(t["text"].split()) < 3:
                turns.append([t["speaker"], t["text"]])
                continue
            try:
                para = llm.structured(_prompt(t["text"]), Para, max_tokens=160).text.strip()
            except LLMError:
                failed += 1
                turns.append([t["speaker"], t["text"]])
                continue
            n0, e0 = _facts(t["text"])
            n1, e1 = _facts(para)
            ok = n0 == n1 and e0 <= e1 and para.lower() != t["text"].lower() and len(para) < 3 * len(t["text"]) + 40
            if ok:
                accepted += 1
                turns.append([t["speaker"], re.sub(r"\s+", " ", para)])
            else:
                rejected += 1
                reasons["numbers_changed" if n0 != n1 else "entity_or_cue_lost" if not e0 <= e1 else "other"] += 1
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
        "llm_usage": {
            "calls": llm.usage.calls,
            "prompt_tokens": llm.usage.prompt_tokens,
            "completion_tokens": llm.usage.completion_tokens,
            "seconds": llm.usage.seconds,
        },
    }
