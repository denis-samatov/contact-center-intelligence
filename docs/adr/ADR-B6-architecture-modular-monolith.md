# ADR-B6: Modular monolith with two process types; Airflow only for the nightly batch

Status: accepted (2026-10-22)

## Decision
One Python package (`cci`) with layers `domain` (schema, numerals, routing rules - no I/O),
`speech`/`nlp`/`rag` (models), `app` (API, DB, worker) and `eval`. Deployed as two process
types from one image: `cci serve` (stateless API) and `cci worker` (models, CPU-heavy), plus
PostgreSQL and the llama.cpp server. Airflow runs the nightly DAG (report, feedback export,
retrain + gate) because those are periodic, need retries/backfill, and touch the registry.

## Not done
Kubernetes manifests are not shipped: there is no cluster in the authoring environment, so
they could not be validated. The two process types, health/readiness probes and graceful
SIGTERM handling are what a Deployment would need.
