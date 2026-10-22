# Runbook

## Start

```bash
export CCI_DATABASE_URL=postgresql+psycopg://cci:cci@localhost:5432/cci   # or sqlite:///./cci.db
export CCI_API_KEY=...                         # required; requests without it get 401
export CCI_INTENT_MODEL_URI=models:/cci-intent@champion   # or a local .joblib
export CCI_LLM_BASE_URL=http://127.0.0.1:8081/v1          # optional (KB answers, LLM paths)
export CCI_EMBED_BASE_URL=http://127.0.0.1:8082/v1        # optional (dense retrieval)
cci serve      # API + UI on :8090
cci worker     # one or more; claims jobs with SKIP LOCKED
```
Or `CCI_API_KEY=... docker compose up` (db + api + worker verified; the llama.cpp services were
not built in the development environment).

Health: `GET /health/live` (process up), `GET /health/ready` (database reachable; reports
whether KB answers are available). Metrics: `GET /metrics` (`cci_jobs_queued`, `cci_jobs_failed`).
Worker logs are one JSON object per line (`job_claimed`, `job_done`, `job_failed`).

## Job lifecycle

`queued -> running -> done | failed`. A failure is retried with backoff `2^attempt` seconds up
to `max_attempts` (3). A worker that dies mid-job leaves the job `running`; another worker
re-claims it after `CCI_JOB_VISIBILITY_TIMEOUT_S` (900 s). SIGTERM finishes the current job,
then the worker exits. Results of a call are written in one transaction, so a re-processed
call never ends up with half a case.

## Symptoms

| symptom | check | action |
|---|---|---|
| `cci_jobs_queued` grows | number of workers; `job_done` seconds in logs (~16 s per 50 s call on 4 cores) | add workers (each loads the ASR and diarization models; check its RSS before sizing); ASR is CPU-bound |
| `cci_jobs_failed` grows | `job_failed` errors; `calls.error` | `FileNotFoundError` for models -> run `scripts/fetch_models.sh`; audio decode errors -> the upload is not a supported WAV |
| jobs stuck in `running` | `jobs.locked_at`, worker alive? | wait for the visibility timeout or restart workers; do not edit rows by hand |
| `/health/ready` 503 | database | the API stays up and reports not ready; fix the database, no restart needed |
| KB answers missing (`kb_answers: false`) | `CCI_LLM_BASE_URL`, llama-server health | the rest of the pipeline works without the LLM; answers return when the server is back |
| many cases sent to review | `GET /v1/analytics` review reasons | expected with ASR errors on numbers and product names; compare with `results/rules_v1_1.json` before changing thresholds |
| 401 / 422 on upload | key; provenance | provenance must be `synthetic`, `licensed` or `consented_recording` - by design |
| 429 | client sends faster than `CCI_RATE_LIMIT_PER_MIN` | back off per `Retry-After`; raise the limit only for trusted keys |
| new intent model not used | MLflow registry alias `champion` | `cci train-intent` promotes only if dev macro-F1 is not lower than the champion's by more than 0.01; check the version's `gate` tag |
| Airflow nightly DAG failed | task logs (`daily_report`, `feedback_dataset`, `retrain_gate`) | tasks are idempotent per `ds`; clear and re-run the failed task |

## Release and rollback

* Code: new image tag; DB changes are additive, so the previous image runs on the new schema.
* Intent model: re-point the `champion` alias to the previous registry version; each case
  stores the model versions it was produced with (`case.model_versions`), so affected cases can be found.
* Rules: version string `RULES_VERSION` in `src/cci/nlp/rules.py`; any change is re-scored with
  `cci eval-rules` on dev before release.
