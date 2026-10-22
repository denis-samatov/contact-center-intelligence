"""Sensitivity grid for the business-case model in docs/BUSINESS_CASE.md.

Measured inputs (automation rate, error rate on automated) are read from results/nlp.json;
everything else is an explicit assumption passed on the command line.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def net_hours(calls: int, a: float, e: float, s: float, r: float, v: float, b: float) -> float:
    return calls * (a * s - a * e * r - (1 - a) * max(0.0, v - b)) / 60


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--calls", type=int, default=10000, help="ASSUMPTION: calls per month")
    p.add_argument("--acw", type=float, default=3.0, help="ASSUMPTION: today's ACW minutes per call (b)")
    p.add_argument("--review", type=float, default=1.0, help="ASSUMPTION: minutes to review a pre-filled case (v)")
    p.add_argument(
        "--rework", type=float, default=10.0, help="ASSUMPTION: minutes of rework per wrong automated case (r)"
    )
    p.add_argument("--results", default="results/nlp.json")
    a = p.parse_args()
    sel = json.loads(Path(a.results).read_text())["selective"]["test_selective"]
    auto, err = sel["automation_rate"], sel["error_rate_on_automated"] or 0.0
    print(f"MEASURED on synthetic test split: automation_rate={auto:.3f}, error_rate_on_automated={err:.3f}")
    print(f"ASSUMED: calls={a.calls}/month, ACW={a.acw} min, review={a.review} min, rework={a.rework} min\n")
    print("ACW saved per automated call (s) | net hours saved / month at measured (a, e) | at a-0.2, e*2 (pessimistic)")
    for s in (0.5, 1.0, 2.0, a.acw):
        base = net_hours(a.calls, auto, err, s, a.rework, a.review, a.acw)
        pess = net_hours(a.calls, max(0.0, auto - 0.2), min(1.0, err * 2), s, a.rework, a.review, a.acw)
        print(f"{s:>32.1f} | {base:>40.1f} | {pess:>8.1f}")


if __name__ == "__main__":
    main()
