"""Operational analytics over processed calls: repeat contacts, queue volumes, operator load,
automation and review statistics. Pure functions over row dicts (the API feeds them from
PostgreSQL; the batch DAG feeds them from the day's partition)."""

from __future__ import annotations

import datetime as dt
from collections import Counter, defaultdict


def repeat_contacts(rows: list[dict], window_days: int = 7) -> dict:
    """A contact is a repeat if the same policy or claim number called within the window."""
    by_key: dict[str, list[dt.date]] = defaultdict(list)
    for r in rows:
        key = r["fields"].get("policy_number") or r["fields"].get("claim_number")
        if key:
            by_key[key].append(dt.date.fromisoformat(r["call_date"]))
    repeats = 0
    for dates in by_key.values():
        dates.sort()
        repeats += sum(1 for a, b in zip(dates, dates[1:], strict=False) if (b - a).days <= window_days)
    identified = sum(len(v) for v in by_key.values())
    return {
        "identified_contacts": identified,
        "repeat_contacts": repeats,
        "repeat_rate": repeats / identified if identified else 0.0,
        "window_days": window_days,
    }


def queue_volumes(rows: list[dict]) -> dict[str, int]:
    return dict(Counter(r["queue"] for r in rows).most_common())


def operator_load(rows: list[dict]) -> list[dict]:
    """Talk time per operator from diarized speech durations."""
    agg: dict[str, dict] = defaultdict(lambda: {"calls": 0, "operator_talk_s": 0.0, "client_talk_s": 0.0})
    for r in rows:
        a = agg[r.get("operator", "unknown")]
        a["calls"] += 1
        a["operator_talk_s"] += r.get("operator_talk_s", 0.0)
        a["client_talk_s"] += r.get("client_talk_s", 0.0)
    out = []
    for op, a in sorted(agg.items()):
        total = a["operator_talk_s"] + a["client_talk_s"]
        out.append(
            {
                "operator": op,
                **a,
                "mean_handle_s": total / a["calls"] if a["calls"] else 0.0,
                "operator_talk_share": a["operator_talk_s"] / total if total else 0.0,
            }
        )
    return out


def automation(rows: list[dict]) -> dict:
    n = len(rows)
    auto = [r for r in rows if r["decision"] == "auto"]
    reasons = Counter(x for r in rows for x in r.get("review_reasons", []))
    return {
        "calls": n,
        "automated": len(auto),
        "automation_rate": len(auto) / n if n else 0.0,
        "review_reasons": dict(reasons.most_common()),
    }


def daily_report(rows: list[dict]) -> dict:
    return {
        "automation": automation(rows),
        "queues": queue_volumes(rows),
        "repeats": repeat_contacts(rows),
        "operators": operator_load(rows),
        "intents": dict(Counter(r["intent"] for r in rows).most_common()),
    }
