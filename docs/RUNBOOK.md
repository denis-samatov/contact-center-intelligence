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

## Claim generations and authoritative corrections

Workers renew `locked_at` while processing. Every result or retry transaction checks the
running job's worker, attempt generation and live visibility window in the same transaction.
A stale worker cannot complete/requeue a replacement claim. A CPU/model call already in
progress may finish after ownership loss, but its commit is rejected; no forced thread kill
is attempted. Reviewed cases and their transcripts are retained when a job is reprocessed.
Do not manually alter job ownership to force concurrent work.

Corrections always reassemble schema fields, contradictions, missing information, queue,
next actions, protocol and reply sources through the shared domain builder. A changed
intent has no inherited classifier confidence. Explicit human approved/rejected decisions
are recorded as such; otherwise changed inputs are re-evaluated for review. The expected
case version is checked by a conditional write as well as the initial read.

## Before accepting real recordings

The current single API key is a trusted operator/integration credential; `reviewer` is an
operator-supplied audit label, not independently authenticated human identity. Do not expose
it as a multi-user customer interface. Provenance/consent labels are declarations, not proof
that recording/data rights have been obtained.

Before an authorized pilot, document owner-approved recording rights, role/subject mapping,
retention and deletion schedules (original audio, transcripts, case/review records, backups
and failure logs), storage access/encryption and reviewer accountability. Validate failures
without logging customer speech or credentials. These governance choices require the data
owner and target identity system; a synthetic test cannot establish them.

Acceptance in isolated approved staging:

1. Pin model/data source hashes and package resolution; use only trusted MLflow/joblib
   artifacts. Re-run the no-model API/worker suite and the separate installed Airflow job.
2. On PostgreSQL, run two workers across short visibility windows, process death and
   deliberate slow processing; assert exactly one owner can commit and human corrections
   survive. Repeat optimistic corrections from two clients and expect a 409 conflict.
3. Select an untouched, consented target-domain holdout before tuning: spontaneous speech,
   speaker overlaps, accents, channel noise and difficult numbers. Evaluate ASR, intent,
   fields, contradictions, review coverage and accepted-case error with appropriate clusters.
4. Back up/restore database, audio storage and pinned model/registry identities together;
   verify retry/review versions, result completeness and reference hashes before resume.
5. Exercise the owner-approved retention/deletion procedure and production access controls
   in staging before ingesting customer data. Agree SLOs and monitor reviewed outcomes.

No new real-call/GPU/live-service/restore qualification is implied by the code fixes.
Historical `results/` remain reports of their original synthetic experiments.

Queue retry exhaustion also covers hard process death: once an expired claim has used
`max_attempts`, the next poll atomically marks the job and call failed and clears the
lease. It never grants attempt `max_attempts + 1`. The stale CPU worker remains fenced.
