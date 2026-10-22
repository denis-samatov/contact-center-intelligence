# Architecture

```mermaid
flowchart LR
  UP[Upload + provenance] --> API[FastAPI /v1] --> PG[(PostgreSQL:<br/>calls, segments, cases,<br/>reviews, jobs)]
  WK[Worker] -->|SKIP LOCKED claim| PG
  WK --> VAD[Silero VAD] --> ASR[GigaAM-v2 / Parakeet / Whisper<br/>sherpa-onnx]
  WK --> DZ[pyannote seg + ERes2Net<br/>diarization] --> ROLES[operator/client roles]
  ASR --> NORM[number normalization]
  NORM --> INT[intent: TF-IDF / embeddings / LLM]
  NORM --> EXT[extraction: rules / LLM]
  INT --> CASE[case: required fields, contradictions,<br/>queue, next actions, review decision]
  EXT --> CASE --> PG
  OP[Operator UI] -->|corrections, versioned| API
  KB[(rulebook chunks)] --> RAG[BM25 + Qwen3-Embedding, RRF] --> ANS[grounded answer + verification]
  AF[Airflow nightly] --> REP[report] & FB[feedback export] & TR[retrain + gate -> MLflow registry]
```

## Key design choices
* Speech and NLP stages are plain functions over typed data (`pipeline.py`), so evaluation
  runs the same code as the worker.
* Russian numerals are normalized deterministically (`domain/ru_numbers.py`): ASR emits
  words, extraction needs digits; dictated groups ("сто двадцать три четыреста пятьдесят
  шесть") are kept as digit strings.
* The case decision is explainable: each review reason is an observable signal.
* Models are versioned in the MLflow registry (`cci-intent@champion`); the worker records the
  model versions it used on every case.

## Trade-offs
* CPU-only ONNX inference keeps the stack portable; GPU would cut ASR latency but is not
  needed for batch post-call processing at contact-center volumes.
* Rules vs LLM extraction is decided by measurement, per condition (see results).
