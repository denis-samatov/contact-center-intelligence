"""HTTP API (FastAPI) and operator UI.

The API stores uploads and enqueues jobs; heavy models run in the worker. Endpoints are
versioned under ``/v1``. Authentication: ``X-API-Key`` (constant-time comparison);
anonymous access only when explicitly enabled *and* the client is on loopback.
"""

from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import tempfile
import time
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, update

from cci.app.db import CallRow, CaseRow, JobRow, ReviewRow, SegmentRow, db_ready, init_db, make_engine
from cci.app.ratelimit import EXEMPT_PREFIXES, TokenBucketLimiter, client_id, retry_after
from cci.app.settings import ALLOWED_PROVENANCE, Settings

UI = Path(__file__).with_name("ui.html")
AUDIO_TYPES = {".wav", ".flac", ".ogg"}


class Metrics:
    def __init__(self):
        self.counters: dict[tuple[str, str], int] = {}
        self.latency_sum: dict[str, float] = {}

    def observe(self, route: str, status: int, seconds: float) -> None:
        k = (route, str(status))
        self.counters[k] = self.counters.get(k, 0) + 1
        self.latency_sum[route] = self.latency_sum.get(route, 0.0) + seconds

    def render(self, extra: dict[str, float]) -> str:
        lines = ["# TYPE cci_http_requests_total counter"]
        lines += [
            f'cci_http_requests_total{{route="{r}",status="{s}"}} {n}' for (r, s), n in sorted(self.counters.items())
        ]
        lines.append("# TYPE cci_http_request_seconds_sum counter")
        lines += [f'cci_http_request_seconds_sum{{route="{r}"}} {v:.6f}' for r, v in sorted(self.latency_sum.items())]
        for name, v in extra.items():
            lines += [f"# TYPE {name} gauge", f"{name} {v}"]
        return "\n".join(lines) + "\n"


class CaseCorrection(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    intent: str | None = None
    fields: dict[str, str] | None = None
    decision: str | None = Field(None, pattern="^(approved|rejected|review)$")
    comment: str = Field("", max_length=2000)
    expected_version: int = Field(ge=1, description="optimistic concurrency: the version the reviewer saw")


class Question(BaseModel):
    question: str = Field(min_length=3, max_length=500)


def create_app(settings: Settings | None = None, answerer=None, limiter: TokenBucketLimiter | None = None) -> FastAPI:
    s = settings or Settings()
    limiter = limiter or TokenBucketLimiter(s.rate_limit_per_min)
    engine = make_engine(s.database_url)
    Session = init_db(engine)
    s.storage_dir.mkdir(parents=True, exist_ok=True)
    metrics = Metrics()
    app = FastAPI(title="Contact Center Intelligence", version="0.1.0")

    @app.middleware("http")
    async def _observe(request: Request, call_next):
        t0 = time.perf_counter()
        response = await call_next(request)
        route = request.scope.get("route")
        metrics.observe(getattr(route, "path", "unmatched"), response.status_code, time.perf_counter() - t0)
        return response

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if request.url.path.startswith(EXEMPT_PREFIXES):
            return await call_next(request)
        host = request.client.host if request.client else None
        wait = limiter.take(client_id(request.headers.get("x-api-key"), host))
        if wait:
            return JSONResponse(
                {"detail": "rate limit exceeded"}, status_code=429, headers={"Retry-After": retry_after(wait)}
            )
        return await call_next(request)

    def auth(request: Request) -> str:
        key = request.headers.get("x-api-key", "")
        if s.api_key is not None and hmac.compare_digest(key.encode(), s.api_key.get_secret_value().encode()):
            return "api-key"
        host = request.client.host if request.client else ""
        if s.allow_anonymous_local and host in ("127.0.0.1", "::1", "testclient"):
            return "anonymous-local"
        raise HTTPException(401, "missing or invalid API key")

    @app.get("/health/live")
    def live():
        return {"status": "ok"}

    @app.get("/health/ready")
    def ready():
        ok = db_ready(engine)
        if not ok:
            raise HTTPException(503, "database unavailable")
        return {"status": "ready", "kb_answers": answerer is not None}

    @app.get("/metrics", response_class=PlainTextResponse)
    def prom():
        with Session() as session:
            queued = session.query(JobRow).filter(JobRow.status == "queued").count()
            failed = session.query(JobRow).filter(JobRow.status == "failed").count()
        return metrics.render({"cci_jobs_queued": queued, "cci_jobs_failed": failed})

    @app.get("/", response_class=HTMLResponse)
    def ui():
        return UI.read_text(encoding="utf-8")

    @app.post("/v1/calls", status_code=202)
    async def upload(
        file: UploadFile = File(...),
        provenance: str = Form(...),
        license_note: str = Form(""),
        call_date: dt.date | None = Form(None),
        who: str = Depends(auth),
    ):
        if provenance not in ALLOWED_PROVENANCE:
            raise HTTPException(
                422,
                f"provenance must be one of {ALLOWED_PROVENANCE}; recordings of real people require documented consent",
            )
        suffix = Path(file.filename or "").suffix.lower()
        if suffix not in AUDIO_TYPES:
            raise HTTPException(415, f"unsupported audio type {suffix!r}; use {sorted(AUDIO_TYPES)}")
        data = await file.read(s.max_upload_mb * 2**20 + 1)
        if len(data) > s.max_upload_mb * 2**20:
            raise HTTPException(413, "file too large")
        digest = hashlib.sha256(data).hexdigest()
        path = s.storage_dir / f"{digest}{suffix}"
        if not path.exists():
            with tempfile.NamedTemporaryFile(dir=s.storage_dir, suffix=".part", delete=False) as tmp:
                temporary = Path(tmp.name)
                tmp.write(data)
            try:
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
        with Session() as session:
            call = CallRow(
                filename=Path(file.filename).name,
                sha256=digest,
                storage_path=str(path),
                provenance=provenance,
                license_note=license_note,
                call_date=call_date.isoformat() if call_date else None,
            )
            session.add(call)
            session.flush()
            session.add(JobRow(call_id=call.id))
            session.commit()
            return {"call_id": call.id, "status": call.status, "sha256": digest, "uploaded_by": who}

    @app.get("/v1/calls")
    def list_calls(
        status: str | None = None,
        decision: str | None = None,
        queue: str | None = None,
        limit: int = Query(50, ge=1),
        _: str = Depends(auth),
    ):
        with Session() as session:
            q = select(CallRow, CaseRow).join(CaseRow, CaseRow.call_id == CallRow.id, isouter=True)
            if status:
                q = q.where(CallRow.status == status)
            if decision:
                q = q.where(CaseRow.decision == decision)
            if queue:
                q = q.where(CaseRow.queue == queue)
            rows = session.execute(q.order_by(CallRow.uploaded_at.desc()).limit(min(limit, 500))).all()
            return [
                {
                    "call_id": c.id,
                    "filename": c.filename,
                    "status": c.status,
                    "provenance": c.provenance,
                    "intent": k.intent if k else None,
                    "decision": k.decision if k else None,
                    "queue": k.queue if k else None,
                }
                for c, k in rows
            ]

    @app.get("/v1/calls/{call_id}")
    def get_call(call_id: str, _: str = Depends(auth)):
        with Session() as session:
            call = session.get(CallRow, call_id)
            if call is None:
                raise HTTPException(404, "call not found")
            segs = session.scalars(
                select(SegmentRow).where(SegmentRow.call_id == call_id).order_by(SegmentRow.idx)
            ).all()
            case = session.get(CaseRow, call_id)
            reviews = session.scalars(select(ReviewRow).where(ReviewRow.call_id == call_id)).all()
            return {
                "call_id": call.id,
                "filename": call.filename,
                "status": call.status,
                "error": call.error,
                "provenance": call.provenance,
                "duration_s": call.duration_s,
                "asr_backend": call.asr_backend,
                "timings": call.timings,
                "segments": [
                    {
                        "start": x.start,
                        "end": x.end,
                        "role": x.role,
                        "speaker": x.speaker,
                        "text": x.text,
                        "text_normalized": x.text_normalized,
                    }
                    for x in segs
                ],
                "case": None
                if case is None
                else {
                    k: getattr(case, k)
                    for k in (
                        "version",
                        "intent",
                        "intent_prob",
                        "fields",
                        "missing",
                        "contradictions",
                        "preferences",
                        "queue",
                        "next_actions",
                        "decision",
                        "review_reasons",
                        "protocol",
                        "reply_draft",
                        "reply_sources",
                        "model_versions",
                        "reviewed",
                    )
                },
                "reviews": [
                    {
                        "reviewer": r.reviewer,
                        "after": r.after,
                        "comment": r.comment,
                        "created_at": r.created_at.isoformat(),
                    }
                    for r in reviews
                ],
            }

    @app.patch("/v1/calls/{call_id}/case")
    def correct_case(call_id: str, body: CaseCorrection, _: str = Depends(auth)):
        from cci.domain.schema import FIELDS, Contradiction, Intent
        from cci.nlp.case import ReviewPolicy, build_case, render_protocol
        from cci.nlp.rules import Extraction

        with Session() as session:
            case = session.get(CaseRow, call_id, with_for_update=engine.dialect.name == "postgresql")
            if case is None:
                raise HTTPException(404, "case not found")
            if case.version != body.expected_version:
                raise HTTPException(409, f"case changed (version {case.version}); reload before editing")
            before = {"intent": case.intent, "fields": dict(case.fields), "decision": case.decision}
            try:
                intent = Intent(body.intent if body.intent is not None else case.intent)
            except ValueError as e:
                raise HTTPException(422, f"unknown intent {body.intent!r}") from e
            fields = {**case.fields, **(body.fields or {})}
            unknown = set(body.fields or {}) - set(FIELDS)
            if unknown:
                raise HTTPException(422, f"unknown fields {sorted(unknown)}")
            corrected = set(body.fields or {})
            derived = build_case(
                intent.value,
                case.intent_prob if intent.value == case.intent else 0.0,
                Extraction(
                    fields={k: v for k, v in fields.items() if v},
                    preferences=case.preferences,
                    contradictions=[
                        Contradiction.model_validate(c) for c in case.contradictions if c["field"] not in corrected
                    ],
                ),
                ReviewPolicy(min_intent_prob=s.min_intent_prob),
            )
            if body.decision is not None:
                derived.decision = body.decision
            elif body.intent is None and body.fields is None:
                derived.decision = case.decision
            derived.protocol = render_protocol(derived)
            values = {
                name: getattr(derived, name)
                for name in (
                    "intent",
                    "intent_prob",
                    "fields",
                    "missing",
                    "contradictions",
                    "queue",
                    "next_actions",
                    "decision",
                    "review_reasons",
                    "protocol",
                    "reply_draft",
                    "reply_sources",
                )
            }
            result = session.execute(
                update(CaseRow)
                .where(CaseRow.call_id == call_id, CaseRow.version == body.expected_version)
                .values(**values, version=body.expected_version + 1, reviewed=True)
            )
            if result.rowcount != 1:
                raise HTTPException(409, "case changed; reload before editing")
            session.refresh(case)
            after = {"intent": case.intent, "fields": dict(case.fields), "decision": case.decision}
            session.add(
                ReviewRow(call_id=call_id, reviewer=body.reviewer, before=before, after=after, comment=body.comment)
            )
            session.commit()
            return {"call_id": call_id, "version": case.version, **after, "missing": case.missing, "queue": case.queue}

    @app.post("/v1/kb/answer")
    def kb_answer(q: Question, _: str = Depends(auth)):
        if answerer is None:
            raise HTTPException(503, "knowledge-base answers need a configured LLM (CCI_LLM_BASE_URL)")
        return answerer(q.question)

    @app.get("/v1/analytics")
    def analytics(_: str = Depends(auth)):
        from cci.analytics.metrics import daily_report

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
                    "operator": "operator",
                    "operator_talk_s": c.operator_talk_s or 0.0,
                    "client_talk_s": c.client_talk_s or 0.0,
                }
                for c, k in rows
            ]
        return json.loads(json.dumps(daily_report(data)))

    @app.get("/v1/dashboard")
    def dashboard(_: str = Depends(auth)):
        from cci.analytics.dashboard import offline_quality, operations

        with Session() as session:
            calls = [
                {"status": c.status, "duration_s": c.duration_s, "timings": c.timings}
                for c in session.scalars(select(CallRow))
            ]
        ops = operations(calls, analytics("dashboard"))
        return json.loads(json.dumps({"operations": ops, "offline_quality": offline_quality(s.results_dir)}))

    return app
