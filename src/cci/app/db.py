"""Persistence: SQLAlchemy 2 models for PostgreSQL (SQLite is supported for unit tests).

The job table is a durable work queue: workers claim rows with
``SELECT ... FOR UPDATE SKIP LOCKED`` (PostgreSQL), so several workers can run safely and a
crashed worker's job is re-queued after its visibility timeout. Keeping jobs in the same
database as call state makes "state change + job completion" one transaction
(``docs/adr/ADR-B2-postgres-job-queue.md``).
"""

from __future__ import annotations

import datetime as dt
import uuid

from sqlalchemy import JSON, DateTime, Float, ForeignKey, Integer, String, Text, create_engine, select, text
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, sessionmaker


def _now() -> dt.datetime:
    return dt.datetime.now(dt.UTC)


class Base(DeclarativeBase):
    pass


class CallRow(Base):
    __tablename__ = "calls"
    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255))
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    storage_path: Mapped[str] = mapped_column(Text)
    provenance: Mapped[str] = mapped_column(String(32))  # synthetic | licensed | consented_recording
    license_note: Mapped[str] = mapped_column(Text, default="")
    call_date: Mapped[str | None] = mapped_column(String(10), nullable=True)
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    duration_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    asr_backend: Mapped[str | None] = mapped_column(String(32), nullable=True)
    operator_talk_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    client_talk_s: Mapped[float | None] = mapped_column(Float, nullable=True)
    timings: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    uploaded_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class SegmentRow(Base):
    __tablename__ = "segments"
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), primary_key=True)
    idx: Mapped[int] = mapped_column(Integer, primary_key=True)
    start: Mapped[float] = mapped_column(Float)
    end: Mapped[float] = mapped_column(Float)
    speaker: Mapped[str] = mapped_column(String(16))
    role: Mapped[str] = mapped_column(String(16))
    text: Mapped[str] = mapped_column(Text)
    text_normalized: Mapped[str] = mapped_column(Text)


class CaseRow(Base):
    __tablename__ = "cases"
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), primary_key=True)
    version: Mapped[int] = mapped_column(Integer, default=1)
    intent: Mapped[str] = mapped_column(String(32), index=True)
    intent_prob: Mapped[float] = mapped_column(Float)
    fields: Mapped[dict] = mapped_column(JSON)
    missing: Mapped[list] = mapped_column(JSON)
    contradictions: Mapped[list] = mapped_column(JSON)
    preferences: Mapped[dict] = mapped_column(JSON)
    queue: Mapped[str] = mapped_column(String(32), index=True)
    next_actions: Mapped[list] = mapped_column(JSON)
    decision: Mapped[str] = mapped_column(String(16), index=True)
    review_reasons: Mapped[list] = mapped_column(JSON)
    protocol: Mapped[str] = mapped_column(Text)
    reply_draft: Mapped[str] = mapped_column(Text)
    reply_sources: Mapped[list] = mapped_column(JSON)
    model_versions: Mapped[dict] = mapped_column(JSON, default=dict)
    reviewed: Mapped[bool] = mapped_column(default=False)
    updated_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now, onupdate=_now)


class ReviewRow(Base):
    """Human corrections: an append-only audit trail and the feedback dataset for retraining."""

    __tablename__ = "reviews"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    before: Mapped[dict] = mapped_column(JSON)
    after: Mapped[dict] = mapped_column(JSON)
    comment: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)


class JobRow(Base):
    __tablename__ = "jobs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    call_id: Mapped[str] = mapped_column(ForeignKey("calls.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(32), default="process_call")
    status: Mapped[str] = mapped_column(String(16), default="queued", index=True)  # queued|running|done|failed
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    max_attempts: Mapped[int] = mapped_column(Integer, default=3)
    locked_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    locked_at: Mapped[dt.datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    run_after: Mapped[dt.datetime] = mapped_column(DateTime(timezone=True), default=_now)
    last_error: Mapped[str | None] = mapped_column(Text, nullable=True)


def make_engine(url: str):
    kwargs = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def init_db(engine) -> sessionmaker[Session]:
    Base.metadata.create_all(engine)
    return sessionmaker(engine, expire_on_commit=False)


def claim_job(session: Session, worker_id: str, visibility_timeout_s: int = 900) -> JobRow | None:
    """Atomically claim one runnable job (or a job whose worker died)."""
    now = _now()
    stale = now - dt.timedelta(seconds=visibility_timeout_s)
    stmt = (
        select(JobRow)
        .where(
            ((JobRow.status == "queued") & (JobRow.run_after <= now))
            | ((JobRow.status == "running") & (JobRow.locked_at < stale))
        )
        .order_by(JobRow.id)
        .limit(1)
    )
    if session.bind.dialect.name == "postgresql":
        stmt = stmt.with_for_update(skip_locked=True)
    job = session.scalars(stmt).first()
    if job is None:
        return None
    job.status, job.locked_by, job.locked_at = "running", worker_id, now
    job.attempts += 1
    session.commit()
    return job


def db_ready(engine) -> bool:
    try:
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:  # noqa: BLE001
        return False
