"""End-to-end processing of one call: audio -> transcript -> case.

Stages (each timed; failures in optional stages degrade gracefully and are recorded):

1. ASR with VAD segmentation and timestamps.
2. Diarization -> speaker per segment -> operator/client roles.
3. Transcript normalization (numbers to digits for display; raw text kept).
4. Intent classification (with probability).
5. Extraction: rules (default, deterministic) or LLM (``extractor="llm"``).
6. Case assembly: required fields, contradictions, queue, next actions, review decision,
   protocol and reply draft.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from cci.domain.ru_numbers import normalize_numbers
from cci.nlp.case import Case, ReviewPolicy, build_case
from cci.nlp.rules import extract as rule_extract


@dataclass
class TranscriptSegment:
    start: float
    end: float
    speaker: str
    role: str
    text: str
    text_normalized: str


@dataclass
class CallResult:
    call_id: str
    asr_backend: str
    duration_s: float
    segments: list[TranscriptSegment]
    case: Case
    timings_s: dict[str, float] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)


class CallProcessor:
    def __init__(
        self,
        transcriber,
        diarizer,
        intent_model,
        policy: ReviewPolicy | None = None,
        extractor: str = "rules",
        llm=None,
    ):
        self.transcriber = transcriber
        self.diarizer = diarizer
        self.intent_model = intent_model
        self.policy = policy or ReviewPolicy()
        self.extractor = extractor
        self.llm = llm

    def process(self, call_id: str, samples: np.ndarray, call_date: dt.date | None = None) -> CallResult:
        from cci.speech.diarization import assign_roles, assign_speakers

        timings, warnings = {}, []
        t0 = time.perf_counter()
        asr = self.transcriber.transcribe(samples)
        timings["asr"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        if self.diarizer is not None:
            try:
                turns = self.diarizer(samples)
                speakers = assign_speakers(asr.segments, turns)
            except Exception as e:  # noqa: BLE001 - diarization is optional; record and continue
                warnings.append(f"diarization_failed: {e}")
                speakers = ["spk0"] * len(asr.segments)
        else:
            speakers = ["spk0"] * len(asr.segments)
        roles = assign_roles(asr.segments, speakers)
        timings["diarization"] = time.perf_counter() - t0
        segs = [
            TranscriptSegment(s.start, s.end, spk, roles.get(spk, "client"), s.text, normalize_numbers(s.text))
            for s, spk in zip(asr.segments, speakers, strict=True)
            if s.text
        ]
        text = " ".join(s.text for s in segs)
        t0 = time.perf_counter()
        labels, probs = self.intent_model.predict_proba([text])
        timings["intent"] = time.perf_counter() - t0
        t0 = time.perf_counter()
        turns_rt = [(s.role, s.text) for s in segs]
        if self.extractor == "llm" and self.llm is not None:
            from cci.nlp.llm_extract import llm_extract

            ex = llm_extract(self.llm, turns_rt, call_date)
        else:
            ex = rule_extract(turns_rt, call_date)
        timings["extraction"] = time.perf_counter() - t0
        case = build_case(labels[0], float(probs[0].max()), ex, self.policy)
        return CallResult(call_id, asr.backend, asr.audio_s, segs, case, timings, warnings)


def load_processor(
    models_dir: Path, intent_model, asr_backend: str = "gigaam-v2", diarize: bool = True, num_threads: int = 4
) -> CallProcessor:
    from cci.speech.asr import Transcriber
    from cci.speech.diarization import Diarizer

    return CallProcessor(
        Transcriber(asr_backend, models_dir, num_threads),
        Diarizer(models_dir, num_threads=num_threads) if diarize else None,
        intent_model,
    )
