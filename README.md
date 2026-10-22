# contact-center-intelligence

**From a recorded insurance call to a verified case card: Russian ASR with timestamps,
speaker roles, structured fields, contradictions, protocol, grounded answers with sources,
and an explicit automate-or-review decision.**

| | |
|---|---|
| **Problem** | Operators spend after-call work typing summaries and case fields; number errors cause rework and repeat contacts; automation without a confidence signal is unsafe. |
| **For** | Insurance and banking contact centers; their DS/NLP teams. |
| **Data** | 900 **synthetic** calls (9 intents) generated from gold cases and rendered with CC0 TTS voices; family-disjoint train/dev/test phrasings; telephony and noise conditions. No real customer data. |
| **Stack** | sherpa-onnx (GigaAM-v2, Parakeet-TDT-0.6B-v3, Whisper-small, Silero VAD, pyannote-seg + ERes2Net diarization), scikit-learn, Qwen3-1.7B and Qwen3-Embedding-0.6B via llama.cpp (local, OpenAI-compatible API), FastAPI, PostgreSQL job queue, MLflow tracking + registry, Airflow 3 nightly DAG. |

## Run

```bash
python -m venv .venv && . .venv/bin/activate && pip install -e '.[dev]'
pytest -q                                   # unit, API, queue, retries (no models needed)
scripts/fetch_models.sh models              # ~1 GB ONNX speech models
cci corpus && cci render --split test --condition telephony --limit 5
cci eval-asr --split test --condition telephony --limit 5 --diarization
cci train-intent                            # MLflow run + registry alias "champion"
CCI_API_KEY=demo CCI_INTENT_MODEL_URI=models/intent/intent.joblib cci serve & cci worker
```
Open `http://127.0.0.1:8090`, paste the key, upload a WAV with provenance `synthetic`.

## Measured results (CPU, synthetic speech)

Full report with CIs: [results/REPORT.md](results/REPORT.md). Protocol: [docs/EVALUATION.md](docs/EVALUATION.md).

**ASR, 60 test calls (34 min of audio), WER with numbers normalized on both sides**

| condition | GigaAM-v2 | Parakeet-TDT-0.6B-v3 | Whisper-small |
|---|---:|---:|---:|
| clean | **9.2%** (8.3-10.1) | 14.6% (13.2-16.1) | 29.7% (27.8-31.7) |
| telephony (8 kHz, mu-law) | **9.3%** (8.4-10.2) | 16.4% (14.5-18.3) | 31.6% (29.5-33.5) |
| telephony + noise 10 dB | **9.4%** (8.3-10.4) | 23.1% (20.7-25.5) | not run |
| real-time factor (telephony, 4 CPU threads) | 0.14 | 0.11 | 0.40 |

GigaAM-v2 (Russian-specific) is the default: best WER in every condition and robust to the
telephone channel and noise. Diarization on telephony audio: DER 0.108; see the report for
role-assignment accuracy and the fix developed on dev.

<!-- NLP_RESULTS -->

## What the product does (and where)

| Function | Code |
|---|---|
| Upload with declared provenance; SHA-256 dedup; durable job | `src/cci/app/api.py`, `app/db.py` |
| VAD, ASR with timestamps, diarization, roles | `src/cci/speech/` |
| Russian numeral normalization (words <-> digits, dictated groups) | `src/cci/domain/ru_numbers.py` |
| Intent: TF-IDF / embeddings / LLM zero-shot | `src/cci/nlp/intent.py` |
| Extraction: rules with corrections & contradictions / LLM JSON schema | `src/cci/nlp/rules.py`, `nlp/llm_extract.py` |
| Case: required fields, queue, next actions, review reasons, protocol, reply draft with sources | `src/cci/nlp/case.py` |
| Grounded answers with citation and numeric verification | `src/cci/rag/` |
| Operator corrections with optimistic locking, audit trail | `PATCH /v1/calls/{id}/case` |
| Analytics: repeat contacts, queues, operator load, automation | `src/cci/analytics/metrics.py`, `GET /v1/analytics` |
| MLflow registry with quality gate; Airflow nightly DAG | `src/cci/training.py`, `airflow/dags/cci_nightly.py` |

## Responsible use

Uploads require provenance `synthetic`, `licensed` or `consented_recording`. Routing and
recommendations use only the stated intent and explicitly stated preferences; no emotion,
demographic or health inference; no automated insurance decisions ([ADR-B4](docs/adr/ADR-B4-no-sensitive-inference.md)).
TTS voices were selected by dataset license ([ADR-B3](docs/adr/ADR-B3-tts-voice-licensing.md)).

## Limits

* Synthetic TTS speech: no overlaps, accents or spontaneous disfluency. WER here validates the
  pipeline and ranks the models under identical conditions; it is not production accuracy.
* Rules were written against the generator's grammar; their reference-text scores are an upper
  bound. Robustness is measured on ASR and LLM-paraphrase conditions.
* Qwen3-ASR could not be evaluated (weights only on Hugging Face, blocked in the build
  environment). One local 1.7B LLM; larger models not evaluated.

[PRD](docs/PRD.md) · [Architecture](docs/ARCHITECTURE.md) · [Business case](docs/BUSINESS_CASE.md) ·
[Evaluation protocol](docs/EVALUATION.md) · [Interview notes](docs/INTERVIEW.md) · [ADRs](docs/adr/) · [Demo script](docs/DEMO.md)
