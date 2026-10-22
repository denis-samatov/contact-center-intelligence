"""API + durable queue behaviour on SQLite. The call processor is a stand-in here because these
tests are about HTTP contracts, auth, retries and transactions; the real speech/NLP pipeline
is exercised in test_pipeline_models.py and in the evaluation runs."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from cci.app.api import create_app
from cci.app.db import JobRow, init_db, make_engine
from cci.app.settings import Settings
from cci.app.worker import Worker
from cci.nlp.case import ReviewPolicy, build_case
from cci.nlp.rules import Extraction
from cci.pipeline import CallResult, TranscriptSegment


def wav_bytes(seconds=0.5):
    buf = io.BytesIO()
    sf.write(buf, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class StubProcessor:
    def __init__(self, fail_times=0):
        self.fail_times, self.calls = fail_times, 0

    def process(self, call_id, samples, call_date=None):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise RuntimeError("transient ASR failure")
        ex = Extraction(fields={"claim_number": "123456"})
        case = build_case("payment_status", 0.95, ex, ReviewPolicy())
        segs = [
            TranscriptSegment(0.0, 1.0, "spk0", "operator", "добрый день", "добрый день"),
            TranscriptSegment(1.2, 2.0, "spk1", "client", "статус выплаты", "статус выплаты"),
        ]
        return CallResult(call_id, "stub", 2.0, segs, case, {"asr": 0.01})


@pytest.fixture()
def env(tmp_path):
    s = Settings(database_url=f"sqlite:///{tmp_path}/t.db", storage_dir=tmp_path / "st", api_key="secret-key")
    return s, TestClient(create_app(s))


H = {"X-API-Key": "secret-key"}


def upload(client, provenance="synthetic", name="a.wav", headers=H):
    return client.post(
        "/v1/calls",
        headers=headers,
        files={"file": (name, wav_bytes(), "audio/wav")},
        data={"provenance": provenance, "license_note": "generated", "call_date": "2026-10-10"},
    )


def test_auth_required_and_constant_time_key(env):
    _, c = env
    assert upload(c, headers={}).status_code == 401
    assert upload(c, headers={"X-API-Key": "wrong"}).status_code == 401
    assert upload(c).status_code == 202


def test_provenance_and_type_validation(env):
    _, c = env
    assert upload(c, provenance="scraped").status_code == 422
    assert upload(c, name="a.exe").status_code == 415


def test_end_to_end_queue_and_correction(env):
    s, c = env
    call_id = upload(c).json()["call_id"]
    w = Worker(s, StubProcessor(), {"intent": "test"})
    assert w.run_once() is True and w.run_once() is False
    d = c.get(f"/v1/calls/{call_id}", headers=H).json()
    assert d["status"] == "done" and d["case"]["intent"] == "payment_status" and len(d["segments"]) == 2
    # human correction with optimistic concurrency
    r = c.patch(
        f"/v1/calls/{call_id}/case",
        headers=H,
        json={
            "reviewer": "op1",
            "intent": "complaint",
            "fields": {"complaint_subject": "delay"},
            "decision": "approved",
            "expected_version": 1,
        },
    )
    assert r.status_code == 200 and r.json()["queue"] == "complaints" and r.json()["version"] == 2
    stale = c.patch(f"/v1/calls/{call_id}/case", headers=H, json={"reviewer": "op2", "expected_version": 1})
    assert stale.status_code == 409
    bad = c.patch(
        f"/v1/calls/{call_id}/case", headers=H, json={"reviewer": "op2", "fields": {"ssn": "x"}, "expected_version": 2}
    )
    assert bad.status_code == 422
    d = c.get(f"/v1/calls/{call_id}", headers=H).json()
    assert d["case"]["reviewed"] and d["reviews"][0]["reviewer"] == "op1"
    assert c.get("/v1/analytics", headers=H).json()["automation"]["calls"] == 1


def test_retry_with_backoff_then_success(env):
    s, c = env
    call_id = upload(c).json()["call_id"]
    w = Worker(s, StubProcessor(fail_times=1), {})
    w.run_once()  # fails -> re-queued with run_after in the future
    Session = init_db(make_engine(s.database_url))
    with Session() as session:
        job = session.query(JobRow).one()
        assert job.status == "queued" and job.attempts == 1 and "transient" in job.last_error
        job.run_after = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)
        session.commit()
    w.run_once()
    assert c.get(f"/v1/calls/{call_id}", headers=H).json()["status"] == "done"


def test_permanent_failure_after_max_attempts(env):
    s, c = env
    call_id = upload(c).json()["call_id"]
    w = Worker(s, StubProcessor(fail_times=99), {})
    Session = init_db(make_engine(s.database_url))
    for _ in range(3):
        w.run_once()
        with Session() as session:
            for job in session.query(JobRow):
                job.run_after = dt.datetime.now(dt.UTC) - dt.timedelta(seconds=1)
            session.commit()
    d = c.get(f"/v1/calls/{call_id}", headers=H).json()
    assert d["status"] == "failed" and "transient" in d["error"]


def test_stale_running_job_is_reclaimed(env):
    s, c = env
    upload(c)
    Session = init_db(make_engine(s.database_url))
    with Session() as session:  # simulate a worker that crashed after claiming
        job = session.query(JobRow).one()
        job.status, job.locked_by = "running", "dead-worker"
        job.locked_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
        session.commit()
    assert Worker(s, StubProcessor(), {}).run_once() is True


def test_health_and_metrics(env):
    _, c = env
    assert c.get("/health/live").json()["status"] == "ok"
    assert c.get("/health/ready").status_code == 200
    assert "cci_jobs_queued" in c.get("/metrics").text
    assert c.post("/v1/kb/answer", headers=H, json={"question": "Какая франшиза?"}).status_code == 503


def test_rate_limit_returns_429_per_key_and_exempts_probes(tmp_path):
    from cci.app.ratelimit import TokenBucketLimiter

    s = Settings(database_url=f"sqlite:///{tmp_path}/t.db", storage_dir=tmp_path / "st", api_key="secret-key")
    frozen = TokenBucketLimiter(per_minute=2, clock=lambda: 0.0)
    c = TestClient(create_app(s, limiter=frozen))
    h = {"X-API-Key": "secret-key"}
    assert [c.get("/v1/calls", headers=h).status_code for _ in range(2)] == [200, 200]
    r = c.get("/v1/calls", headers=h)
    assert r.status_code == 429 and int(r.headers["Retry-After"]) >= 1
    assert c.get("/v1/calls", headers={"X-API-Key": "guess"}).status_code == 401  # separate bucket
    assert all(c.get("/health/live").status_code == 200 for _ in range(5))
