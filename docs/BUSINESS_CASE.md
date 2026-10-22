# Business case (model with explicit assumptions)

No real customer, contact center or financial result is involved. **Measured** inputs come
from `results/nlp.json` on synthetic data; every other input is an **assumption** to be
replaced with the operator's own data.

## Model

Per month, with automation rate `a` and error rate `e` on automated cases:

    net_hours_saved = calls x [ a x s  -  a x e x r  -  (1 - a) x max(0, v - b) ] / 60
    net_cost_saved  = net_hours_saved x loaded_cost_per_hour - infra_cost_per_month

The first term is ACW saved on automated calls, the second is rework caused by wrong
automated cases, the third is extra time if reviewing a pre-filled case is slower than
today's manual ACW (usually it is faster, then the term is zero). `scripts/business_case.py`
prints a sensitivity grid instead of a single number.

| symbol | meaning | source |
|---|---|---|
| `a` | automation rate | **measured** (selective policy, synthetic test split) |
| `e` | error rate on automated cases | **measured** |
| `calls` | calls per month | assumption |
| `b` | today's ACW minutes per call | assumption |
| `s` | ACW minutes saved on an automated call (<= b) | assumption |
| `v` | minutes to review a pre-filled case | assumption |
| `r` | minutes of rework per wrong automated case | assumption |

The script prints a sensitivity grid instead of one number. A real business case needs a
pilot on consented recordings to replace `a` and `e`: synthetic automation and error
rates are optimistic because TTS audio is clean.
