# ADR-B2: PostgreSQL as the durable job queue (no Redis/Celery for calls)

Status: accepted (2026-10-22)

## Decision
Jobs live in a `jobs` table; workers claim them with `SELECT ... FOR UPDATE SKIP LOCKED`,
retry with exponential backoff, mark permanent failure after `max_attempts`, and reclaim
jobs whose worker died (visibility timeout). Results (segments, case, call status, job
status) are written in **one transaction**.

## Alternatives
* Redis + RQ/Celery: fast, familiar, but job state and case state live in two stores; a
  crash between "write case" and "ack job" needs idempotency logic on both sides.
* Airflow for per-call processing: scheduling latency and per-task overhead are wrong for
  near-real-time work; Airflow is used for the nightly batch instead (ADR-B6).
* Kafka: justified at much higher volumes or with multiple consumers of call events.

## Consequences
+ Exactly-once *effects* per call within the database; simple operations.
- Throughput bounded by PostgreSQL row locking (fine for thousands of calls/hour).
