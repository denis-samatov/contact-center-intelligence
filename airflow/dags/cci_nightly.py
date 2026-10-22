"""Nightly batch pipeline for the contact center (Apache Airflow 3).

Why a scheduled batch exists next to the always-on worker: per-call processing is online
(worker), but three tasks are inherently periodic and benefit from orchestration (retries,
backfills, lineage of daily partitions):

1. ``daily_report``     - analytics over the day's processed calls (repeat contacts, queue
                           volumes, operator load, automation and review reasons).
2. ``feedback_dataset`` - export human corrections (reviews) as training data.
3. ``retrain_gate``     - retrain the intent model on train + feedback, evaluate on the fixed
                           dev set and register it in MLflow; promote only if the quality gate
                           passes (``cci.training.train_and_register``).

Configuration comes from the same ``CCI_*`` environment variables as the services.
Verified locally with ``dag.test()`` against SQLite (see ``tests/test_airflow_dag.py``).
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pendulum
from airflow.sdk import dag, task

OUT = Path(os.environ.get("CCI_BATCH_OUT", "./batch"))


@dag(
    schedule="15 2 * * *",
    start_date=pendulum.datetime(2026, 10, 1, tz="UTC"),
    catchup=False,
    default_args={"retries": 2, "retry_delay": pendulum.duration(minutes=5)},
    tags=["cci"],
)
def cci_nightly():
    @task
    def daily_report(ds: str | None = None) -> str:
        from sqlalchemy import select

        from cci.analytics.metrics import daily_report as report
        from cci.app.db import CallRow, CaseRow, init_db, make_engine
        from cci.app.settings import Settings

        s = Settings()
        Session = init_db(make_engine(s.database_url))
        with Session() as session:
            rows = session.execute(select(CallRow, CaseRow).join(CaseRow, CaseRow.call_id == CallRow.id)).all()
            data = [
                {
                    "call_date": c.call_date or c.uploaded_at.date().isoformat(),
                    "fields": k.fields,
                    "queue": k.queue,
                    "intent": k.intent,
                    "decision": k.decision,
                    "review_reasons": k.review_reasons,
                    "operator_talk_s": c.operator_talk_s or 0.0,
                    "client_talk_s": c.client_talk_s or 0.0,
                }
                for c, k in rows
            ]
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"report-{ds}.json"
        path.write_text(json.dumps(report(data), ensure_ascii=False, indent=1))
        return str(path)

    @task
    def feedback_dataset(ds: str | None = None) -> str:
        from sqlalchemy import select

        from cci.app.db import CaseRow, ReviewRow, SegmentRow, init_db, make_engine
        from cci.app.settings import Settings

        s = Settings()
        Session = init_db(make_engine(s.database_url))
        rows = []
        with Session() as session:
            for case in session.scalars(select(CaseRow).where(CaseRow.reviewed.is_(True))):
                segs = session.scalars(
                    select(SegmentRow).where(SegmentRow.call_id == case.call_id).order_by(SegmentRow.idx)
                ).all()
                last = session.scalars(
                    select(ReviewRow).where(ReviewRow.call_id == case.call_id).order_by(ReviewRow.id.desc())
                ).first()
                if last is not None and last.after.get("decision") != "rejected":
                    rows.append(
                        {"call_id": case.call_id, "intent": case.intent, "text": " ".join(x.text for x in segs)}
                    )
        OUT.mkdir(parents=True, exist_ok=True)
        path = OUT / f"feedback-{ds}.jsonl"
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        return str(path)

    @task
    def retrain_gate(feedback_path: str) -> dict:
        from cci.synth.generator import load_corpus
        from cci.training import feedback_items, train_and_register

        corpus = Path(os.environ.get("CCI_CORPUS", "data/corpus/calls.jsonl"))
        calls = load_corpus(corpus)

        def item(c):
            return {"text_reference": " ".join(t["text"] for t in c["turns"]), "label": c["label"]}

        train = [item(c) for c in calls if c["split"] == "train"]
        dev = [item(c) for c in calls if c["split"] == "dev"]
        fb = [json.loads(line) for line in Path(feedback_path).read_text().splitlines() if line.strip()]
        return train_and_register(
            train,
            dev,
            os.environ.get("CCI_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"),
            extra_train=feedback_items(fb),
            run_name="nightly-retrain",
        )

    daily_report()
    retrain_gate(feedback_dataset())


cci_nightly_dag = cci_nightly()
