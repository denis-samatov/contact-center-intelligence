"""ASR and diarization evaluation on rendered synthetic calls.

For every backend and acoustic condition: corpus WER/CER (edit distance summed over calls,
divided by reference length), with and without numeric normalization, real-time factor,
and per-call bootstrap CI of WER. Diarization: DER (no collar) and role accuracy (share of
reference speech time whose ASR segment got the correct operator/client role).
"""

from __future__ import annotations

import json
from pathlib import Path

from cci.eval.metrics import asr_normalize, bootstrap_ci, cer_counts, norm_text, wer_counts
from cci.speech.asr import Transcriber, load_wav
from cci.speech.diarization import Diarizer, Turn, assign_roles, assign_speakers, der


def _ref_text(ref: dict) -> str:
    return " ".join(s["text"] for s in ref["segments"])


def eval_asr(
    audio_dir: Path, models_dir: Path, backends: list[str], limit: int | None = None, num_threads: int = 4
) -> dict:
    refs = sorted(audio_dir.glob("*.ref.json"))[:limit]
    out = {"audio_dir": str(audio_dir), "calls": len(refs), "backends": {}}
    for backend in backends:
        tr = Transcriber(backend, models_dir, num_threads)
        per_call, tot = (
            [],
            {"w_err": 0, "w_n": 0, "c_err": 0, "c_n": 0, "raw_err": 0, "raw_n": 0, "audio_s": 0.0, "decode_s": 0.0},
        )
        for rp in refs:
            ref = json.loads(rp.read_text(encoding="utf-8"))
            res = tr.transcribe(load_wav(rp.with_name(ref["call_id"] + ".wav")))
            r_norm, h_norm = asr_normalize(_ref_text(ref)), asr_normalize(res.text)
            we, wn = wer_counts(r_norm, h_norm)
            ce, cn = cer_counts(r_norm, h_norm)
            re_, rn = wer_counts(norm_text(_ref_text(ref)), norm_text(res.text))
            tot["w_err"] += we
            tot["w_n"] += wn
            tot["c_err"] += ce
            tot["c_n"] += cn
            tot["raw_err"] += re_
            tot["raw_n"] += rn
            tot["audio_s"] += res.audio_s
            tot["decode_s"] += res.decode_s
            per_call.append(
                {
                    "call_id": ref["call_id"],
                    "wer": we / max(1, wn),
                    "err": we,
                    "n": wn,
                    "hyp": res.text,
                    "ref": _ref_text(ref),
                }
            )
        errs, ns = [p["err"] for p in per_call], [p["n"] for p in per_call]

        def corpus_wer(idx, errs=errs, ns=ns):
            return sum(errs[i] for i in idx) / max(1, sum(ns[i] for i in idx))

        lo, hi = bootstrap_ci(errs, stat=corpus_wer)
        out["backends"][backend] = {
            "wer": tot["w_err"] / tot["w_n"],
            "wer_ci95": [lo, hi],
            "cer": tot["c_err"] / tot["c_n"],
            "wer_without_number_normalization": tot["raw_err"] / tot["raw_n"],
            "rtf": tot["decode_s"] / tot["audio_s"],
            "audio_s": tot["audio_s"],
            "examples": per_call[:3],
            "worst": sorted(per_call, key=lambda p: -p["wer"])[:3],
        }
    return out


def eval_diarization(
    audio_dir: Path, models_dir: Path, asr_backend: str = "gigaam-v2", limit: int | None = None, num_threads: int = 4
) -> dict:
    refs = sorted(audio_dir.glob("*.ref.json"))[:limit]
    tr = Transcriber(asr_backend, models_dir, num_threads)
    dz = Diarizer(models_dir, num_threads=num_threads)
    rows = []
    for rp in refs:
        ref = json.loads(rp.read_text(encoding="utf-8"))
        x = load_wav(rp.with_name(ref["call_id"] + ".wav"))
        turns = dz(x)
        ref_turns = [Turn(s["speaker"], s["start"], s["end"]) for s in ref["segments"]]
        d = der(ref_turns, turns)
        asr = tr.transcribe(x)
        speakers = assign_speakers(asr.segments, turns)
        roles = assign_roles(asr.segments, speakers)
        correct = total = 0.0
        for seg, spk in zip(asr.segments, speakers, strict=True):
            for rs in ref["segments"]:
                ov = max(0.0, min(seg.end, rs["end"]) - max(seg.start, rs["start"]))
                total += ov
                if roles.get(spk) == rs["speaker"]:
                    correct += ov
        rows.append(
            {
                "call_id": ref["call_id"],
                **d,
                "role_accuracy": correct / total if total else 0.0,
                "n_hyp_speakers": len({t.speaker for t in turns}),
            }
        )
    n = len(rows)
    return {
        "calls": n,
        "der_mean": sum(r["der"] for r in rows) / n,
        "confusion_mean": sum(r["confusion"] for r in rows) / n,
        "missed_mean": sum(r["missed"] for r in rows) / n,
        "false_alarm_mean": sum(r["false_alarm"] for r in rows) / n,
        "role_accuracy_mean": sum(r["role_accuracy"] for r in rows) / n,
        "calls_with_role_accuracy_below_0.9": sum(r["role_accuracy"] < 0.9 for r in rows),
        "rows": rows,
    }
