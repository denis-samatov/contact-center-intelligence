# Product requirements: Contact Center Intelligence (insurance)

## Audience
Contact-center operations and claims teams of insurers and banks; their Data Science / NLP
team that owns the models; supervisors who review automated cases.

## Problem
After each call an operator spends post-call work time (after-call work, ACW) writing a
summary, filling the case form (policy number, dates, places, amounts) and routing it.
Quality varies, mistakes in numbers cause rework and repeat contacts, and supervisors can't
see why customers call back. Speech analytics products exist, but they either do not
extract structured, verifiable case data or they automate without telling when they are
unsure.

## Workflow
1. Recording is uploaded with a declared **provenance** (synthetic / licensed /
   consented recording). Anything else is rejected.
2. Worker: VAD -> ASR (GigaAM-v2 by default) with timestamps -> diarization -> operator /
   client roles -> number normalization.
3. Intent classification with a probability; rule-based (default) or LLM extraction of the
   intent's fields; corrections handled; contradictions and missing required fields found.
4. Case card: queue (deterministic routing rules), next actions, protocol, reply draft with
   KB sources, decision **auto** vs **review** with explicit reasons.
5. Operator reviews in the UI, corrects fields; corrections are versioned (optimistic
   locking) and become training data.
6. Nightly batch (Airflow): analytics report, feedback export, retrain + quality gate +
   MLflow registry promotion.
7. Operators ask policy questions; answers are grounded in the rulebook with citations and
   are verified (citations valid, numbers supported) before being shown as confirmed.

## Constraints
* No real personal data in the public demo; all calls are synthetic.
* No inference of sensitive or psychological attributes; routing and recommendations use
  only the stated intent and explicitly stated preferences (ADR-B4).
* Runs on CPU; external LLM providers only if explicitly configured.
* Every automated decision is explainable by observable signals (probability, missing
  fields, contradictions).

## Competing approaches
Manual ACW; commercial speech analytics (keyword spotting + dashboards); generic
"LLM summarizes the call" tools without verification; IVR menus. This product sits between:
structured extraction + selective automation with measured error on automated cases.

## Acceptance criteria
| # | Criterion | Evidence |
|---|---|---|
| 1 | Upload a permitted audio file and get transcript with timestamps and roles, intent, fields, protocol, sources, and correct it in the UI | `tests/test_api_worker.py`, screenshots in `docs/demo/` |
| 2 | Two or more ASR candidates compared with WER/CER + CI on the same audio | `results/asr_test_*.json` |
| 3 | Classical baseline vs neural/LLM for intent and extraction | `results/nlp.json` |
| 4 | Retrieval-free vs grounded answers compared | `results/rag.json` |
| 5 | Automate-all vs selective prediction with threshold chosen on dev | `results/nlp.json` (`selective`) |
| 6 | MLflow tracking + registry with quality gate; Airflow DAG tested | `src/cci/training.py`, `tests/test_airflow_dag.py` |
| 7 | Failure handling: retries, permanent failure, stale-job reclaim | `tests/test_api_worker.py` |
