# ADR-B9: localdesk as an evaluated intent model; template-cluster confidence intervals

Status: accepted (2026-10-22).

## Context
`localdesk` (github.com/denis-samatov/localdesk, PyPI 0.3.1) is the author's calibrated text
classifier with abstention (`route` / `review` from calibrated confidence and top-2 margin),
quality gates and an immutable registry. This project had its own TF-IDF + logistic regression
and an intent-probability threshold - functionally overlapping. The question was whether to
replace them with localdesk.

Integrating it exposed a property of the corpus: localdesk refused a random train/calibration
split because identical inputs crossed splits. The 600 train calls contain only **54 distinct
first utterances** (6 per intent); dev and test have **27 each** (3 per intent). Calls that share
an opening template are not independent observations.

## Decision
* localdesk is integrated through its own pipeline (`src/cci/nlp/localdesk_intent.py`): task
  spec with label descriptions, explicit splits with SHA-256 fingerprints, train / calibration
  split by localdesk's input-equivalence groups (stratified per label), its gates and registry;
  our dev split is its reporting split; thresholds are its defaults, fixed before test.
* It is **evaluated, not made the default**: on test (ASR openings) it reaches 0.467 accuracy
  vs 0.580 for our TF-IDF + LR and 0.800 for Qwen3-Embedding + LR, and at its default thresholds
  it sends every test call to review (its calibrated confidence never reaches 0.75). That is
  correct abstaining behaviour for a weak model, but it automates nothing here.
* Early-routing metrics get **template-cluster bootstrap CIs** (resample templates, not calls).
  They are 2-2.5x wider: embeddings + LR 0.800 [0.652, 0.925] vs TF-IDF 0.580 [0.388, 0.768] on
  ASR openings. The intervals overlap; with 27 templates the 0.22 gap is large but not established.

## Consequences
* The overlap with localdesk is now a measured comparison instead of a duplicated component.
  With a realistic training set (many distinct phrasings) localdesk's calibrated abstention is
  the natural selective-prediction layer; on 54 distinct openings it is not.
* Section 2 of the report keeps its call-level intervals for continuity and points to section
  4a; README and résumé material use the template-level intervals.
* The data generator should produce more distinct openings per intent (roadmap): the corpus,
  not the models, limits how precisely early routing can be measured.
