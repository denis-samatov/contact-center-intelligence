# ADR-B1: Synthetic calls with family-disjoint splits and a fresh test seed

Status: accepted (2026-10-22)

## Context
No labelled Russian insurance calls with permission for public use are available (and real
calls contain personal data). Hugging Face and OpenSLR are not reachable from the build
environment, so even public Russian speech corpora could not be used.

## Decision
Generate dialogues from structured cases (gold labels by construction), render them with
CC0 Piper voices, and evaluate under three conditions: reference text, ASR on telephony-
degraded audio, and LLM paraphrases of client turns with fact verification. Client openings
come from four template families: train {A,B}, dev {C}, test {D}. Rules and thresholds are
tuned on dev only. Because an early debugging run looked at a sample of the first test seed,
the reported corpus uses a new seed (20261023) generated after the rules were frozen.

## Consequences
+ Exact gold labels for intent, fields, contradictions, missing fields, preferences, queue.
- Results measure the pipeline on synthetic speech; they are not production accuracy.
  Rule extraction on reference text is near-perfect by construction and is labelled so.
