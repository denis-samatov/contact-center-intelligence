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


def test_dashboard_separates_operations_from_offline_quality(tmp_path):
    import json as _json

    res = tmp_path / "results"
    res.mkdir()
    (res / "rag.json").write_text(
        _json.dumps({"grounded_fully_correct": 0.8, "closed_book_fully_correct": 0.05, "citation_precision": 0.9})
    )
    s = Settings(
        database_url=f"sqlite:///{tmp_path}/t.db", storage_dir=tmp_path / "st", api_key="secret-key", results_dir=res
    )
    c = TestClient(create_app(s))
    d = c.get("/v1/dashboard", headers={"X-API-Key": "secret-key"}).json()
    assert d["operations"]["calls_by_status"] == {} and d["operations"]["processing_s"]["n"] == 0
    assert {r["metric"]: r["value"] for r in d["offline_quality"]}["fully correct, grounded"] == 0.8
    assert all(r["source"] == "results/rag.json" for r in d["offline_quality"])  # missing files are skipped
    assert c.get("/v1/dashboard").status_code == 401


def test_stale_worker_cannot_complete_a_reclaimed_job(env):
    from cci.app.db import CaseRow

    settings, client = env
    call_id = upload(client).json()["call_id"]
    Session = init_db(make_engine(settings.database_url))

    class Reclaimed(StubProcessor):
        def process(self, *args):
            result = super().process(*args)
            with Session() as session:
                job = session.query(JobRow).one()
                job.locked_by = "replacement-worker"
                job.attempts += 1
                session.commit()
            return result

    assert Worker(settings, Reclaimed(), {}).run_once()
    with Session() as session:
        job = session.query(JobRow).one()
        assert job.status == "running" and job.locked_by == "replacement-worker"
        assert session.get(CaseRow, call_id) is None


def test_stale_failure_cannot_requeue_another_workers_claim(env):
    settings, client = env
    upload(client)
    Session = init_db(make_engine(settings.database_url))

    class ReclaimedFailure(StubProcessor):
        def process(self, *args):
            with Session() as session:
                job = session.query(JobRow).one()
                job.locked_by = "replacement-worker"
                job.attempts += 1
                session.commit()
            raise RuntimeError("old attempt failed")

    assert Worker(settings, ReclaimedFailure(), {}).run_once()
    with Session() as session:
        job = session.query(JobRow).one()
        assert job.status == "running" and job.locked_by == "replacement-worker"


def test_reprocessing_does_not_overwrite_a_reviewed_case(env):
    from dataclasses import asdict

    from cci.app.db import CaseRow

    settings, client = env
    call_id = upload(client).json()["call_id"]
    Session = init_db(make_engine(settings.database_url))

    class ReviewedDuringProcessing(StubProcessor):
        def process(self, *args):
            result = super().process(*args)
            human = asdict(result.case)
            human["fields"] = {"claim_number": "human-corrected"}
            with Session() as session:
                session.add(CaseRow(call_id=call_id, version=7, reviewed=True, model_versions={}, **human))
                session.commit()
            return result

    assert Worker(settings, ReviewedDuringProcessing(), {}).run_once()
    with Session() as session:
        case = session.get(CaseRow, call_id)
        assert case.reviewed and case.version == 7
        assert case.fields == {"claim_number": "human-corrected"}


def test_intent_only_correction_filters_fields_and_rebuilds_case(env):
    settings, client = env
    call_id = upload(client).json()["call_id"]
    assert Worker(settings, StubProcessor(), {}).run_once()
    response = client.patch(
        f"/v1/calls/{call_id}/case",
        headers=H,
        json={"reviewer": "operator", "intent": "general_info", "expected_version": 1},
    )
    assert response.status_code == 200
    case = client.get(f"/v1/calls/{call_id}", headers=H).json()["case"]
    assert case["fields"] == {}
    assert set(case["missing"]) == {"product", "question_topic"}
    assert "Информационный вопрос" in case["protocol"]
    assert case["reply_sources"] == []
    assert case["decision"] == "review"


def test_field_correction_rebuilds_missing_actions_and_protocol(env):
    settings, client = env
    call_id = upload(client).json()["call_id"]
    assert Worker(settings, StubProcessor(), {}).run_once()
    response = client.patch(
        f"/v1/calls/{call_id}/case",
        headers=H,
        json={"reviewer": "operator", "fields": {"claim_number": ""}, "expected_version": 1},
    )
    assert response.status_code == 200
    case = client.get(f"/v1/calls/{call_id}", headers=H).json()["case"]
    assert case["decision"] == "review" and "missing_required_fields" in case["review_reasons"]
    assert "123456" not in case["protocol"] and "НЕ УКАЗАНО" in case["protocol"]
    assert case["next_actions"] == ["request_missing_info", "send_payment_status"]


def test_invalid_call_date_and_negative_list_limit_are_client_errors(env):
    _, client = env
    response = client.post(
        "/v1/calls",
        headers=H,
        files={"file": ("a.wav", wav_bytes(), "audio/wav")},
        data={"provenance": "synthetic", "call_date": "not-a-date"},
    )
    assert response.status_code == 422
    assert client.get("/v1/calls?limit=-1", headers=H).status_code == 422


def test_heartbeat_prevents_reclaim_during_long_processing(env):
    import time

    from cci.app.db import claim_job

    settings, client = env
    settings.job_visibility_timeout_s = 1
    upload(client)
    Session = init_db(make_engine(settings.database_url))

    class SlowProcessor(StubProcessor):
        def process(self, *args):
            time.sleep(1.2)
            with Session() as session:
                assert claim_job(session, "other-worker", 1) is None
            return super().process(*args)

    assert Worker(settings, SlowProcessor(), {}).run_once()
    with Session() as session:
        assert session.query(JobRow).one().status == "done"


def test_empty_intent_correction_is_not_silently_ignored(env):
    settings, client = env
    call_id = upload(client).json()["call_id"]
    assert Worker(settings, StubProcessor(), {}).run_once()
    response = client.patch(
        f"/v1/calls/{call_id}/case", headers=H, json={"reviewer": "operator", "intent": "", "expected_version": 1}
    )
    assert response.status_code == 422


def test_empty_api_key_cannot_enable_unauthenticated_requests():
    with pytest.raises(ValueError):
        Settings(api_key="")


def test_hard_crash_reclaims_stop_at_max_attempts(env):
    import datetime as dt

    from cci.app.db import CallRow, claim_job

    settings, client = env
    call_id = upload(client).json()["call_id"]
    Session = init_db(make_engine(settings.database_url))
    for attempt in range(1, 4):
        with Session() as session:
            job = claim_job(session, f"crashing-worker-{attempt}", 1)
            assert job.attempts == attempt
            job.locked_at = dt.datetime.now(dt.UTC) - dt.timedelta(hours=1)
            session.commit()
    for _ in range(2):
        with Session() as session:
            assert claim_job(session, "replacement", 1) is None
            job = session.query(JobRow).one()
            assert job.status == "failed" and job.attempts == job.max_attempts == 3
            assert job.locked_by is None and job.locked_at is None
            call = session.get(CallRow, call_id)
            assert call.status == "failed" and "attempt" in call.error
