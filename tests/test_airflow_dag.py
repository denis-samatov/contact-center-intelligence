"""Run the nightly DAG end to end with ``dag.test()`` (needs apache-airflow; separate CI job)."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(importlib.util.find_spec("airflow") is None, reason="apache-airflow not installed")
ROOT = Path(__file__).resolve().parents[1]

SCRIPT = r"""
import json, os, sys
from pathlib import Path
from cci.app.db import CallRow, CaseRow, ReviewRow, SegmentRow, init_db, make_engine
from cci.synth.generator import generate_corpus, write_corpus
tmp = Path(os.environ["T"])
write_corpus(generate_corpus(120, 40, 0, seed=5), tmp / "corpus")
S = init_db(make_engine(os.environ["CCI_DATABASE_URL"]))
with S() as s:
    s.add(CallRow(id="c1", filename="a.wav", sha256="x", storage_path="/dev/null", provenance="synthetic",
                  call_date="2026-10-10", status="done", operator_talk_s=20.0, client_talk_s=15.0))
    s.add(CaseRow(call_id="c1", intent="complaint", intent_prob=0.9, fields={"claim_number": "123456"}, missing=[],
                  contradictions=[], preferences={}, queue="complaints", next_actions=[], decision="auto",
                  review_reasons=[], protocol="", reply_draft="", reply_sources=[], reviewed=True))
    s.add(SegmentRow(call_id="c1", idx=0, start=0, end=1, speaker="spk0", role="client", text="хочу оставить жалобу",
                     text_normalized="хочу оставить жалобу"))
    s.add(ReviewRow(call_id="c1", reviewer="op", before={}, after={"decision": "approved"}))
    s.commit()
sys.path.insert(0, os.environ["DAGS"])
from cci_nightly import cci_nightly_dag
run = cci_nightly_dag.test(logical_date=None)
print("STATE", run.state)
"""


def test_nightly_dag_runs_end_to_end(tmp_path):
    env = {
        **os.environ,
        "T": str(tmp_path),
        "AIRFLOW_HOME": str(tmp_path / "airflow"),
        "AIRFLOW__CORE__LOAD_EXAMPLES": "False",
        "AIRFLOW__DATABASE__SQL_ALCHEMY_CONN": f"sqlite:///{tmp_path}/af.db",
        "CCI_DATABASE_URL": f"sqlite:///{tmp_path}/cci.db",
        "CCI_BATCH_OUT": str(tmp_path / "batch"),
        "CCI_CORPUS": str(tmp_path / "corpus" / "calls.jsonl"),
        "CCI_MLFLOW_TRACKING_URI": f"sqlite:///{tmp_path}/mlflow.db",
        "DAGS": str(ROOT / "airflow" / "dags"),
        "PYTHONPATH": str(ROOT / "src"),
        "MLFLOW_DISABLE_AGENT_HINT": "1",
        "AIRFLOW__CORE__DAGS_FOLDER": str(ROOT / "airflow" / "dags"),
    }
    subprocess.run(
        [sys.executable, "-m", "airflow", "db", "migrate"], env=env, check=True, capture_output=True, timeout=600
    )
    # Airflow 3 creates DAG runs only for serialized DAGs.
    subprocess.run(
        [sys.executable, "-m", "airflow", "dags", "reserialize"], env=env, check=True, capture_output=True, timeout=600
    )
    p = subprocess.run([sys.executable, "-c", SCRIPT], env=env, capture_output=True, text=True, timeout=900)
    assert "STATE success" in p.stdout, (p.stdout[-3000:], p.stderr[-5000:])
    reports = list((tmp_path / "batch").glob("report-*.json"))
    assert reports and json.loads(reports[0].read_text())["automation"]["calls"] == 1
    fb = list((tmp_path / "batch").glob("feedback-*.jsonl"))[0].read_text().splitlines()
    assert json.loads(fb[0])["intent"] == "complaint"
