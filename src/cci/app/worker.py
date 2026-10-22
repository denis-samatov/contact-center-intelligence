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
import time

from sqlalchemy import delete

from cci.app.db import CallRow, CaseRow, JobRow, SegmentRow, claim_job, init_db, make_engine
from cci.app.settings import Settings

log = logging.getLogger("cci.worker")


def _log(event: str, **kw) -> None:
    log.info(json.dumps({"event": event, **kw}, ensure_ascii=False, default=str))


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
            job_id, call_id = job.id, job.call_id
        _log("job_claimed", job=job_id, call=call_id, attempt=job.attempts)
        t0 = time.perf_counter()
        try:
            self._process(call_id)
        except Exception as e:  # noqa: BLE001 - every failure becomes a retry or a failed job
            with self.Session() as session:
                job = session.get(JobRow, job_id)
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
            _log("job_failed", job=job_id, call=call_id, error=str(e)[:300])
            return True
        _log("job_done", job=job_id, call=call_id, seconds=round(time.perf_counter() - t0, 3))
        return True

    def _process(self, call_id: str) -> None:
        from cci.speech.asr import load_wav

        with self.Session() as session:
            call = session.get(CallRow, call_id)
            call.status = "processing"
            session.commit()
            path, call_date = call.storage_path, call.call_date
        samples = load_wav(path)
        result = self.processor.process(call_id, samples, dt.date.fromisoformat(call_date) if call_date else None)
        c = result.case
        with self.Session() as session:  # one transaction: segments + case + call + job
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
                    version=1,
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
            for job in session.query(JobRow).filter(JobRow.call_id == call_id, JobRow.status == "running"):
                job.status = "done"
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
