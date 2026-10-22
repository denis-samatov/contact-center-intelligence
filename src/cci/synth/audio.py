"""Render synthetic calls to audio with Piper (VITS) voices via sherpa-onnx.

Output per call: 16 kHz mono PCM16 WAV plus a reference JSON with speaker turns and their
start/end times (exact, because we place the audio ourselves) and an RTTM file for
diarization scoring. Optional telephony degradation: 8 kHz band-limit, mu-law companding
and additive noise at a target SNR.

Voices: only Piper voices trained on CC0 datasets are used (``dmitri`` = operator,
``denis`` = client; see ``docs/adr/ADR-B3-tts-voice-licensing.md``). ``irina`` (dataset
license "Unknown") and ``ruslan`` (CC BY-NC-SA 4.0) are excluded from published audio.

Limitation recorded in the evaluation report: TTS speech is cleaner and more regular than
real calls (no overlaps, no disfluent prosody). WER on this corpus validates the pipeline;
it is not an estimate of production accuracy.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly

TARGET_SR = 16000
OPERATOR_VOICE = "dmitri"
CLIENT_VOICES = ("denis",)


@dataclass(frozen=True)
class Segment:
    speaker: str
    start: float
    end: float
    text: str


class PiperVoices:
    def __init__(self, models_dir: Path, num_threads: int = 2):
        import sherpa_onnx

        self._tts = {}
        for name in (OPERATOR_VOICE, *CLIENT_VOICES):
            d = models_dir / f"vits-piper-ru_RU-{name}-medium"
            cfg = sherpa_onnx.OfflineTtsConfig(
                model=sherpa_onnx.OfflineTtsModelConfig(
                    vits=sherpa_onnx.OfflineTtsVitsModelConfig(
                        model=str(d / f"ru_RU-{name}-medium.onnx"),
                        tokens=str(d / "tokens.txt"),
                        data_dir=str(d / "espeak-ng-data"),
                    ),
                    num_threads=num_threads,
                ),
                max_num_sentences=1,
            )
            self._tts[name] = sherpa_onnx.OfflineTts(cfg)

    def synth(self, voice: str, text: str, speed: float = 1.0) -> np.ndarray:
        audio = self._tts[voice].generate(text, sid=0, speed=speed)
        x = np.asarray(audio.samples, dtype=np.float32)
        if audio.sample_rate != TARGET_SR:
            g = np.gcd(audio.sample_rate, TARGET_SR)
            x = resample_poly(x, TARGET_SR // g, audio.sample_rate // g).astype(np.float32)
        return x


def degrade(x: np.ndarray, snr_db: float | None, rng: np.random.Generator, telephony: bool = True) -> np.ndarray:
    """Narrowband telephone channel (8 kHz + mu-law) and pink-ish additive noise."""
    y = x
    if telephony:
        y = resample_poly(resample_poly(y, 1, 2), 2, 1)[: len(x)]
        mu = 255.0
        y = np.sign(y) * np.log1p(mu * np.abs(np.clip(y, -1, 1))) / np.log1p(mu)
        y = np.round(y * 127) / 127
        y = np.sign(y) * (np.expm1(np.abs(y) * np.log1p(mu)) / mu)
    if snr_db is not None:
        white = rng.standard_normal(len(y)).astype(np.float32)
        pink = np.cumsum(white)
        pink -= np.convolve(pink, np.ones(400) / 400, mode="same")  # remove drift
        pink /= np.std(pink) + 1e-9
        p_sig = np.mean(y**2) + 1e-12
        y = y + pink * np.sqrt(p_sig / 10 ** (snr_db / 10))
    return np.clip(y, -1, 1).astype(np.float32)


def render_call(
    call: dict, voices: PiperVoices, out_dir: Path, seed: int, snr_db: float | None = None, telephony: bool = False
) -> dict:
    rng = random.Random(seed)
    nrng = np.random.default_rng(seed)
    client_voice = rng.choice(CLIENT_VOICES)
    pieces, segments, t = [], [], 0.3
    pieces.append(np.zeros(int(0.3 * TARGET_SR), dtype=np.float32))
    for turn in call["turns"]:
        voice = OPERATOR_VOICE if turn["speaker"] == "operator" else client_voice
        x = voices.synth(voice, turn["text"], speed=rng.uniform(0.95, 1.1))
        x = x / (np.max(np.abs(x)) + 1e-9) * rng.uniform(0.5, 0.8)
        segments.append(Segment(turn["speaker"], round(t, 3), round(t + len(x) / TARGET_SR, 3), turn["text"]))
        gap = rng.uniform(0.35, 0.9)
        pieces += [x, np.zeros(int(gap * TARGET_SR), dtype=np.float32)]
        t += len(x) / TARGET_SR + gap
    audio = np.concatenate(pieces)
    if telephony or snr_db is not None:
        audio = degrade(audio, snr_db, nrng, telephony=telephony)
    out_dir.mkdir(parents=True, exist_ok=True)
    wav = out_dir / f"{call['call_id']}.wav"
    sf.write(wav, audio, TARGET_SR, subtype="PCM_16")
    ref = {
        "call_id": call["call_id"],
        "duration_s": len(audio) / TARGET_SR,
        "client_voice": client_voice,
        "snr_db": snr_db,
        "telephony": telephony,
        "segments": [s.__dict__ for s in segments],
    }
    (out_dir / f"{call['call_id']}.ref.json").write_text(
        json.dumps(ref, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    with (out_dir / f"{call['call_id']}.rttm").open("w") as fh:
        for s in segments:
            fh.write(
                f"SPEAKER {call['call_id']} 1 {s.start:.3f} {s.end - s.start:.3f} <NA> <NA> {s.speaker} <NA> <NA>\n"
            )
    return ref
