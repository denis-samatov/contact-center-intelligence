"""Speaker diarization (pyannote segmentation-3.0 + speaker embeddings via sherpa-onnx),
speaker-to-segment assignment, operator/client role detection and DER scoring."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from cci.speech.asr import SR, AsrSegment

OPERATOR_CUES = re.compile(r"демо.?страхован|меня зовут|оператор|слушаю вас|чем могу помочь|назовите|подскажите номер")


@dataclass
class Turn:
    speaker: str  # cluster id ("spk0") or role ("operator"/"client") after role assignment
    start: float
    end: float


class Diarizer:
    def __init__(
        self,
        models_dir: Path,
        embedding: str = "3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        num_speakers: int = 2,
        num_threads: int = 4,
    ):
        import sherpa_onnx

        cfg = sherpa_onnx.OfflineSpeakerDiarizationConfig(
            segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
                pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                    model=str(models_dir / "sherpa-onnx-pyannote-segmentation-3-0" / "model.onnx")
                ),
                num_threads=num_threads,
            ),
            embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
                model=str(models_dir / embedding), num_threads=num_threads
            ),
            clustering=sherpa_onnx.FastClusteringConfig(num_clusters=num_speakers),
            min_duration_on=0.3,
            min_duration_off=0.5,
        )
        if not cfg.validate():
            raise ValueError("invalid diarization config (model files missing?)")
        self.sd = sherpa_onnx.OfflineSpeakerDiarization(cfg)

    def __call__(self, samples: np.ndarray) -> list[Turn]:
        result = self.sd.process(samples).sort_by_start_time()
        return [Turn(f"spk{r.speaker}", float(r.start), float(r.end)) for r in result]


def assign_speakers(segments: list[AsrSegment], turns: list[Turn]) -> list[str]:
    """Label each ASR segment with the diarization speaker it overlaps most."""
    labels = []
    for s in segments:
        best, best_ov = "unknown", 0.0
        for t in turns:
            ov = min(s.end, t.end) - max(s.start, t.start)
            if ov > best_ov:
                best, best_ov = t.speaker, ov
        labels.append(best)
    return labels


def assign_roles(segments: list[AsrSegment], speakers: list[str]) -> dict[str, str]:
    """Map cluster ids to operator/client: operator script cues, then who speaks first."""
    score: dict[str, float] = {}
    for s, spk in zip(segments, speakers, strict=True):
        score.setdefault(spk, 0.0)
        score[spk] += len(OPERATOR_CUES.findall(s.text.lower()))
    if speakers:
        score[speakers[0]] = score.get(speakers[0], 0.0) + 0.5  # operators open the call
    ranked = sorted(score, key=score.get, reverse=True)
    roles = {spk: "client" for spk in ranked}
    if ranked:
        roles[ranked[0]] = "operator"
    return roles


def der(reference: list[Turn], hypothesis: list[Turn], step: float = 0.01) -> dict:
    """Frame-based diarization error rate, no collar, best speaker mapping (<= 3 speakers).

    DER = (missed speech + false alarm + speaker confusion) / reference speech.
    """
    from itertools import permutations

    end = max([t.end for t in reference] + [t.end for t in hypothesis] + [0.0])
    n = int(np.ceil(end / step)) + 1

    def frames(turns, names):
        a = np.full(n, -1, dtype=int)
        for t in turns:
            a[int(t.start / step) : int(t.end / step)] = names.index(t.speaker)
        return a

    rnames = sorted({t.speaker for t in reference})
    hnames = sorted({t.speaker for t in hypothesis})
    ref = frames(reference, rnames)
    best = None
    for perm in permutations(range(max(len(rnames), len(hnames))), len(hnames)):
        hyp = frames(hypothesis, hnames)
        mapped = np.where(hyp >= 0, np.array(perm + (0,))[np.clip(hyp, 0, None)], -1)
        speech = ref >= 0
        missed = np.sum(speech & (mapped < 0))
        fa = np.sum(~speech & (mapped >= 0))
        conf = np.sum(speech & (mapped >= 0) & (mapped != ref))
        total = (missed + fa + conf) / max(1, np.sum(speech))
        if best is None or total < best["der"]:
            best = {
                "der": float(total),
                "missed": float(missed / max(1, speech.sum())),
                "false_alarm": float(fa / max(1, speech.sum())),
                "confusion": float(conf / max(1, speech.sum())),
            }
    return best or {"der": 1.0, "missed": 1.0, "false_alarm": 0.0, "confusion": 0.0}


__all__ = ["Diarizer", "Turn", "assign_roles", "assign_speakers", "der", "SR"]
