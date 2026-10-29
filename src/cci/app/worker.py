"""Background worker: claims jobs from PostgreSQL, processes calls, writes results atomically.

Run: ``cci worker``. Retries with exponential backoff up to ``max_attempts``; a worker that
dies mid-job leaves the job ``running`` and another worker re-claims it after the visibility
timeout. SIGTERM finishes the current job and exits (graceful shutdown).
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import os
import signal
import socket
import threading
import time

from sqlalchemy import delete, update

from cci.app.db import CallRow, CaseRow, JobRow, SegmentRow, claim_job, init_db, make_engine
from cci.app.settings import Settings

log = logging.getLogger("cci.worker")


def _log(event: str, **kw) -> None:
    log.info(json.dumps({"event": event, **kw}, ensure_ascii=False, default=str))


class LostJob(RuntimeError):
    """A different claim generation owns this job."""


class Worker:
    def __init__(self, settings: Settings, processor, model_versions: dict):
        self.s = settings
        self.processor = processor
        self.model_versions = model_versions
        self.engine = make_engine(settings.database_url)
        self.Session = init_db(self.engine)
        self.worker_id = f"{socket.gethostname()}-{os.getpid()}"
        self.stopping = False

    def run_once(self) -> bool:
        with self.Session() as session:
            job = claim_job(session, self.worker_id, self.s.job_visibility_timeout_s)
            if job is None:
                return False
        job_id, call_id, attempt = job.id, job.call_id, job.attempts
        _log("job_claimed", job=job_id, call=call_id, attempt=attempt)
        t0 = time.perf_counter()
        stop = threading.Event()

        def heartbeat():
            while not stop.wait(max(0.05, self.s.job_visibility_timeout_s / 3)):
                try:
                    with self.Session() as session:
                        self._owned_job(session, job_id, attempt, renew=True)
                        session.commit()
                except Exception:
                    # The CPU operation cannot be killed safely; its final commit is
                    # still fenced and will be rejected after a takeover/expiry.
                    return

        thread = threading.Thread(target=heartbeat, name="cci-lease-heartbeat", daemon=True)
        thread.start()
        try:
            self._process(call_id, job_id, attempt)
        except LostJob:
            _log("job_ownership_lost", job=job_id, call=call_id, attempt=attempt)
        except Exception as e:  # noqa: BLE001 - every owned failure becomes retry/failed
            try:
                with self.Session() as session:
                    job = self._owned_job(session, job_id, attempt)
                    call = session.get(CallRow, call_id)
                    job.last_error = f"{type(e).__name__}: {e}"[:2000]
                    if job.attempts >= job.max_attempts:
                        job.status, call.status, call.error = "failed", "failed", job.last_error
                    else:
                        job.status = "queued"
                        job.run_after = dt.datetime.now(dt.UTC) + dt.timedelta(seconds=2**job.attempts)
                        call.status = "queued"
                    job.locked_by = job.locked_at = None
                    session.commit()
            except LostJob:
                _log("job_ownership_lost", job=job_id, call=call_id, attempt=attempt)
            else:
                _log("job_failed", job=job_id, call=call_id, error=str(e)[:300])
        else:
            _log("job_done", job=job_id, call=call_id, seconds=round(time.perf_counter() - t0, 3))
        finally:
            stop.set()
            thread.join(timeout=1)
        return True

    def _owned_job(self, session, job_id: int, attempt: int, *, renew: bool = False):
        now = dt.datetime.now(dt.UTC)
        result = session.execute(
            update(JobRow)
            .where(
                JobRow.id == job_id,
                JobRow.status == "running",
                JobRow.locked_by == self.worker_id,
                JobRow.attempts == attempt,
                JobRow.locked_at >= now - dt.timedelta(seconds=self.s.job_visibility_timeout_s),
            )
            .values(locked_at=now if renew else JobRow.locked_at)
            .execution_options(synchronize_session=False)
        )
        if result.rowcount != 1:
            raise LostJob("job lease lost")
        return session.get(JobRow, job_id, populate_existing=True)

    def _process(self, call_id: str, job_id: int, attempt: int) -> None:
        from cci.speech.asr import load_wav

        with self.Session() as session:
            self._owned_job(session, job_id, attempt)
            call = session.get(CallRow, call_id)
            call.status = "processing"
            session.commit()
            path, call_date = call.storage_path, call.call_date
        samples = load_wav(path)
        result = self.processor.process(call_id, samples, dt.date.fromisoformat(call_date) if call_date else None)
        c = result.case
        with self.Session() as session:  # one transaction: segments + case + call + job
            job = self._owned_job(session, job_id, attempt)
            existing = session.get(CaseRow, call_id, with_for_update=self.engine.dialect.name == "postgresql")
            if existing is not None and existing.reviewed:
                job.status = "done"
                job.locked_by = job.locked_at = None
                session.get(CallRow, call_id).status = "done"
                session.commit()
                return  # Human-reviewed fields and associated transcript are authoritative.
            session.execute(delete(SegmentRow).where(SegmentRow.call_id == call_id))
            for i, s in enumerate(result.segments):
                session.add(
                    SegmentRow(
                        call_id=call_id,
                        idx=i,
                        start=s.start,
                        end=s.end,
                        speaker=s.speaker,
                        role=s.role,
                        text=s.text,
                        text_normalized=s.text_normalized,
                    )
                )
            session.merge(
                CaseRow(
                    call_id=call_id,
                    version=existing.version + 1 if existing is not None else 1,
                    intent=c.intent,
                    intent_prob=c.intent_prob,
                    fields=c.fields,
                    missing=c.missing,
                    contradictions=c.contradictions,
                    preferences=c.preferences,
                    queue=c.queue,
                    next_actions=c.next_actions,
                    decision=c.decision,
                    review_reasons=c.review_reasons,
                    protocol=c.protocol,
                    reply_draft=c.reply_draft,
                    reply_sources=c.reply_sources,
                    model_versions=self.model_versions,
                    reviewed=False,
                )
            )
            call = session.get(CallRow, call_id)
            call.status, call.error = "done", None
            call.duration_s, call.asr_backend, call.timings = result.duration_s, result.asr_backend, result.timings_s
            call.operator_talk_s = sum(s.end - s.start for s in result.segments if s.role == "operator")
            call.client_talk_s = sum(s.end - s.start for s in result.segments if s.role == "client")
            job.status = "done"
            job.locked_by = job.locked_at = None
            session.commit()

    def run_forever(self) -> None:
        signal.signal(signal.SIGTERM, lambda *_: setattr(self, "stopping", True))
        _log("worker_started", worker=self.worker_id)
        while not self.stopping:
            if not self.run_once():
                time.sleep(self.s.worker_poll_s)
        _log("worker_stopped", worker=self.worker_id)


def main() -> None:
    from cci.app.models import load_intent_model
    from cci.pipeline import CallProcessor
    from cci.speech.asr import Transcriber
    from cci.speech.diarization import Diarizer

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    s = Settings()
    intent, version = load_intent_model(s.intent_model_uri, s.mlflow_tracking_uri)
    from cci.nlp.case import ReviewPolicy

    role_model = None
    role_path = s.models_dir / "intent" / "roles.joblib"
    if role_path.exists():
        import joblib

        role_model = joblib.load(role_path)
    proc = CallProcessor(
        Transcriber(s.asr_backend, s.models_dir, s.num_threads),
        Diarizer(s.models_dir, num_threads=s.num_threads) if s.diarize else None,
        intent,
        ReviewPolicy(min_intent_prob=s.min_intent_prob),
        role_model=role_model,
    )
    Worker(s, proc, {"intent": version, "asr": s.asr_backend, "extractor": "rules"}).run_forever()
