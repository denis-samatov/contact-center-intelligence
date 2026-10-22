"""Russian ASR backends (sherpa-onnx, CPU) with VAD segmentation and timestamps.

Backends (ONNX exports published as sherpa-onnx GitHub release assets):

* ``gigaam-v2``  - GigaAM-v2 RNN-T (Russian), NeMo transducer export.
* ``parakeet-v3`` - NVIDIA Parakeet-TDT-0.6B-v3 (multilingual incl. Russian), int8.
* ``whisper-small`` / ``whisper-base`` - OpenAI Whisper, language forced to Russian.

Long recordings are segmented with Silero VAD; each speech segment is decoded separately and
keeps absolute start/end times. Transducer backends also return per-token times.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SR = 16000


@dataclass
class AsrSegment:
    start: float
    end: float
    text: str
    tokens: list[str] = field(default_factory=list)
    token_times: list[float] = field(default_factory=list)  # absolute seconds


@dataclass
class AsrResult:
    backend: str
    segments: list[AsrSegment]
    audio_s: float
    decode_s: float

    @property
    def text(self) -> str:
        return " ".join(s.text for s in self.segments if s.text)

    @property
    def rtf(self) -> float:
        return self.decode_s / self.audio_s if self.audio_s else 0.0


BACKENDS = {
    "gigaam-v2": "sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19",
    "parakeet-v3": "sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8",
    "whisper-small": "sherpa-onnx-whisper-small",
    "whisper-base": "sherpa-onnx-whisper-base",
}


def _first(d: Path, pattern: str) -> str:
    matches = sorted(d.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"{d}/{pattern}")
    # prefer int8 exports when both exist (smaller, faster on CPU)
    int8 = [m for m in matches if "int8" in m.name]
    return str((int8 or matches)[0])


def build_recognizer(backend: str, models_dir: Path, num_threads: int = 4):
    import sherpa_onnx

    d = models_dir / BACKENDS[backend]
    if backend.startswith("whisper"):
        size = backend.split("-")[1]
        return sherpa_onnx.OfflineRecognizer.from_whisper(
            encoder=_first(d, f"{size}-encoder*.onnx"),
            decoder=_first(d, f"{size}-decoder*.onnx"),
            tokens=_first(d, f"{size}-tokens.txt"),
            language="ru",
            task="transcribe",
            num_threads=num_threads,
        )
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=_first(d, "encoder*.onnx"),
        decoder=_first(d, "decoder*.onnx"),
        joiner=_first(d, "joiner*.onnx"),
        tokens=str(d / "tokens.txt"),
        model_type="nemo_transducer",
        num_threads=num_threads,
    )


class Vad:
    def __init__(self, model: Path, min_silence: float = 0.25, max_speech: float = 20.0):
        import sherpa_onnx

        cfg = sherpa_onnx.VadModelConfig()
        cfg.silero_vad.model = str(model)
        cfg.silero_vad.min_silence_duration = min_silence
        cfg.silero_vad.min_speech_duration = 0.2
        cfg.silero_vad.max_speech_duration = max_speech
        cfg.sample_rate = SR
        self.cfg = cfg

    def segments(self, samples: np.ndarray) -> list[tuple[float, np.ndarray]]:
        import sherpa_onnx

        vad = sherpa_onnx.VoiceActivityDetector(self.cfg, buffer_size_in_seconds=max(30, len(samples) / SR + 5))
        win = self.cfg.silero_vad.window_size
        out = []
        for i in range(0, len(samples), win):
            vad.accept_waveform(samples[i : i + win])
            while not vad.empty():
                seg = vad.front
                out.append((seg.start / SR, np.asarray(seg.samples, dtype=np.float32)))
                vad.pop()
        vad.flush()
        while not vad.empty():
            seg = vad.front
            out.append((seg.start / SR, np.asarray(seg.samples, dtype=np.float32)))
            vad.pop()
        return out


class Transcriber:
    def __init__(self, backend: str, models_dir: Path, num_threads: int = 4):
        self.backend = backend
        self.recognizer = build_recognizer(backend, models_dir, num_threads)
        self.vad = Vad(models_dir / "silero_vad.onnx")

    def transcribe(self, samples: np.ndarray) -> AsrResult:
        t0 = time.perf_counter()
        segs = self.vad.segments(samples)
        streams = []
        for _start, x in segs:
            s = self.recognizer.create_stream()
            s.accept_waveform(SR, x)
            streams.append(s)
        if streams:
            self.recognizer.decode_streams(streams)
        out = []
        for (start, x), s in zip(segs, streams, strict=True):
            r = s.result
            times = [start + float(t) for t in getattr(r, "timestamps", []) or []]
            out.append(
                AsrSegment(
                    round(start, 3),
                    round(start + len(x) / SR, 3),
                    r.text.strip(),
                    list(getattr(r, "tokens", []) or []),
                    times,
                )
            )
        return AsrResult(self.backend, out, len(samples) / SR, time.perf_counter() - t0)


def load_wav(path: Path) -> np.ndarray:
    import soundfile as sf

    x, sr = sf.read(path, dtype="float32", always_2d=False)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if sr != SR:
        from scipy.signal import resample_poly

        g = np.gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x
