"""Command line interface.

cci corpus      [--train 600 --dev 150 --test 150]   synthetic dialogues + gold labels
cci render      --split dev|test [--condition clean|telephony|noisy]
cci paraphrase  --split test                         LLM-paraphrased client turns (facts verified)
cci eval-asr    --split test --condition clean        WER/CER/RTF for all backends + diarization
cci transcribe  --split dev|test                      cache GigaAM transcripts + roles for NLP eval
cci eval-nlp                                          intent / extraction / selective automation
cci eval-rules                                        rules extractor (current version) on dev + test, post-hoc
cci eval-rag                                          retrieval + closed-book vs grounded answers
cci train-intent                                      train + register in MLflow (quality gate)
cci serve | cci worker
cci report                                            results/REPORT.md
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import zlib
from pathlib import Path

DATA = Path("data")
CORPUS = DATA / "corpus" / "calls.jsonl"
AUDIO = DATA / "audio"
RESULTS = Path("results")
MODELS = Path(os.environ.get("CCI_MODELS_DIR", "models"))
LLM_URL = os.environ.get("CCI_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
EMB_URL = os.environ.get("CCI_EMBED_BASE_URL", "http://127.0.0.1:8082/v1")
CONDITIONS = {"clean": (None, False), "telephony": (None, True), "noisy": (10.0, True)}


def _write(name: str, payload: dict) -> Path:
    from cci.eval.report import manifest

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    path.write_text(
        json.dumps({"suite": name, "manifest": manifest(), **payload}, ensure_ascii=False, indent=1, default=str)
    )
    print(f"wrote {path}")
    return path


def _calls(split: str | None = None) -> list[dict]:
    from cci.synth.generator import load_corpus

    calls = load_corpus(CORPUS)
    return [c for c in calls if split is None or c["split"] == split]


def cmd_corpus(a) -> None:
    from cci.synth.generator import generate_corpus, write_corpus

    calls = generate_corpus(a.train, a.dev, a.test, seed=a.seed)
    print(f"wrote {write_corpus(calls, CORPUS.parent)} ({len(calls)} calls, seed {a.seed})")


def cmd_render(a) -> None:
    from cci.synth.audio import PiperVoices, render_call

    voices = PiperVoices(MODELS, num_threads=a.threads)
    snr, tel = CONDITIONS[a.condition]
    calls = _calls(a.split)[: a.limit]
    out = AUDIO / a.split / a.condition
    for c in calls:  # seed per call id: the same call gets the same voice/pauses in every condition
        render_call(c, voices, out, seed=zlib.crc32(c["call_id"].encode()), snr_db=snr, telephony=tel)
    print(f"rendered {len(calls)} calls into {out}")


def _llm():
    from cci.llm.client import LLMClient, LLMConfig

    return LLMClient(LLMConfig(base_url=LLM_URL, model="qwen3-1.7b"))


def _embedder():
    from cci.llm.client import LLMClient, LLMConfig

    client = LLMClient(LLMConfig(base_url=EMB_URL, model="qwen3-embedding-0.6b"))
    return client.embed


def cmd_paraphrase(a) -> None:
    from cci.synth.paraphrase import paraphrase_calls

    calls = _calls(a.split)[: a.limit]
    # 0.6: below the 5th percentile of similarities observed on 6 dev calls; catches garbage only.
    res = paraphrase_calls(calls, _llm(), embed=_embedder(), min_similarity=0.6)
    path = DATA / "corpus" / f"paraphrase-{a.split}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in res["calls"]), encoding="utf-8")
    _write(f"paraphrase_{a.split}", {k: v for k, v in res.items() if k != "calls"})


def cmd_eval_asr(a) -> None:
    from cci.eval.asr_eval import eval_asr, eval_diarization

    d = AUDIO / a.split / a.condition
    res = {"condition": a.condition, **eval_asr(d, MODELS, a.backends.split(","), a.limit, a.threads)}
    if a.diarization:
        res["diarization"] = eval_diarization(d, MODELS, limit=a.limit, num_threads=a.threads)
    _write(f"asr_{a.split}_{a.condition}", res)


def cmd_transcribe(a) -> None:
    """Cache ASR + diarization outputs (turns with roles) for the NLP evaluation."""
    from cci.speech.asr import Transcriber, load_wav
    from cci.speech.diarization import Diarizer, assign_roles, assign_speakers

    tr, dz = Transcriber("gigaam-v2", MODELS, a.threads), Diarizer(MODELS, num_threads=a.threads)
    rows = []
    for c in _calls(a.split)[: a.limit]:
        x = load_wav(AUDIO / a.split / a.condition / f"{c['call_id']}.wav")
        asr = tr.transcribe(x)
        spk = assign_speakers(asr.segments, dz(x))
        roles = assign_roles(asr.segments, spk)
        rows.append(
            {
                "call_id": c["call_id"],
                # role, text, speaker cluster, start, end: speakers are kept so that role assignment
                # can be re-derived without re-running ASR and diarization.
                "turns": [
                    [roles.get(s, "client"), seg.text, s, seg.start, seg.end]
                    for seg, s in zip(asr.segments, spk, strict=True)
                    if seg.text
                ],
            }
        )
    path = DATA / "corpus" / f"asr-{a.split}-{a.condition}.jsonl"
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    print(f"wrote {path} ({len(rows)} calls)")


def _role_accuracy(turns: list, ref: dict, roles: list[str]) -> float:
    correct = total = 0.0
    for t, role in zip(turns, roles, strict=True):
        for rs in ref["segments"]:
            ov = max(0.0, min(t[4], rs["end"]) - max(t[3], rs["start"]))
            total += ov
            correct += ov if role == rs["speaker"] else 0.0
    return correct / total if total else 0.0


def cmd_roles(a) -> None:
    """Compare script-cue role heuristics with the trained role model on cached transcripts;
    with --apply, rewrite the cached roles using the model (develop on dev, apply to test)."""
    from cci.speech.diarization import RoleModel, assign_roles_model

    train = _calls("train")
    model = RoleModel().fit(
        [t["text"] for c in train for t in c["turns"]], [t["speaker"] for c in train for t in c["turns"]]
    )
    path = DATA / "corpus" / f"asr-{a.split}-{a.condition}.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    old, new = [], []
    for r in rows:
        ref = json.loads((AUDIO / a.split / a.condition / f"{r['call_id']}.ref.json").read_text(encoding="utf-8"))
        turns = r["turns"]
        roles = assign_roles_model([t[1] for t in turns], [t[2] for t in turns], model, mode=a.mode)
        old.append(_role_accuracy(turns, ref, [t[0] for t in turns]))
        new.append(_role_accuracy(turns, ref, roles))
        if a.apply:
            for t, role in zip(turns, roles, strict=True):
                t[0] = role
    res = {
        "split": a.split,
        "calls": len(rows),
        "heuristic": {"mean": sum(old) / len(old), "below_0.9": sum(x < 0.9 for x in old)},
        "role_model": {"mean": sum(new) / len(new), "below_0.9": sum(x < 0.9 for x in new)},
        "applied": a.apply,
        "mode": a.mode,
    }
    print(json.dumps(res, indent=1))
    if a.apply:
        path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")
    _write(f"roles_{a.split}", res)


def _items(split: str, limit: int | None = None, condition: str = "telephony") -> list[dict]:
    calls = _calls(split)[:limit]
    asr = {}
    p = DATA / "corpus" / f"asr-{split}-{condition}.jsonl"
    if p.exists():
        asr = {r["call_id"]: r["turns"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    para = {}
    p = DATA / "corpus" / f"paraphrase-{split}.jsonl"
    if p.exists():
        para = {r["call_id"]: r["turns"] for r in map(json.loads, p.read_text(encoding="utf-8").splitlines())}
    items = []
    for c in calls:
        ref_turns = [(t["speaker"], t["text"]) for t in c["turns"]]
        it = {
            "call_id": c["call_id"],
            "label": c["label"],
            "call_date": c["call_date"],
            "turns_reference": ref_turns,
            "text_reference": " ".join(t for _, t in ref_turns),
        }
        if c["call_id"] in asr:
            it["turns_asr"] = [(x[0], x[1]) for x in asr[c["call_id"]]]
            it["text_asr"] = " ".join(t for _, t in it["turns_asr"])
        if c["call_id"] in para:
            it["turns_paraphrase"] = [tuple(x) for x in para[c["call_id"]]]
            it["text_paraphrase"] = " ".join(t for _, t in it["turns_paraphrase"])
        # "opening": the first client utterance only (early routing). Full transcripts make intent
        # trivial on this corpus because operator questions are intent-specific.
        for cond in ("reference", "asr", "paraphrase"):
            if f"turns_{cond}" in it:
                first = next((t for r, t in it[f"turns_{cond}"] if r == "client"), "")
                it[f"text_{cond}:opening"] = first
        items.append(it)
    return items


def cmd_eval_nlp(a) -> None:
    from cci.eval.nlp_eval import run_extraction, run_intent, run_selective
    from cci.nlp.intent import EmbeddingIntent, LLMIntent, TfidfIntent
    from cci.nlp.llm_extract import llm_extract
    from cci.nlp.rules import extract as rule_extract

    train, test = _items("train"), _items("test", a.limit)
    sub = test[: a.llm_limit]  # LLM methods are slow on CPU: evaluated on a fixed prefix of the test split
    conds = [c for c in ("reference", "asr") if all(f"text_{c}" in it for it in test)]
    sub_conds = conds + (["paraphrase"] if sub and all("text_paraphrase" in it for it in sub) else [])
    print("conditions:", conds, "subset conditions:", sub_conds, "subset n:", len(sub))
    Xtr, ytr = [it["text_reference"] for it in train], [it["label"]["intent"] for it in train]
    Xop = [it["text_reference:opening"] for it in train]

    def op(cs):
        return [f"{c}:opening" for c in cs]

    def tag(res, suffix):
        return {f"{k.split('/')[0]}{suffix}/{k.split('/')[1]}": v for k, v in res.items()}

    tfidf = TfidfIntent().fit(Xtr, ytr)
    tfidf_op = TfidfIntent().fit(Xop, ytr)
    intent_res = run_intent({"tfidf_lr": tfidf}, test, conds) | run_intent({"tfidf_lr": tfidf_op}, test, op(conds))
    intent_res |= tag(run_intent({"tfidf_lr": tfidf_op}, sub, op(sub_conds)), "[subset]")
    ext_res = run_extraction(test, conds, {"rules": rule_extract})
    ext_res |= tag(run_extraction(sub, sub_conds, {"rules": rule_extract}), "[subset]")
    usage = None
    if not a.no_llm:
        llm = _llm()
        emb = EmbeddingIntent(_embedder()).fit(Xop, ytr)
        intent_res |= run_intent({"embedding_lr": emb}, test, op(conds))
        intent_res |= tag(run_intent({"embedding_lr": emb}, sub, op(sub_conds)), "[subset]")
        intent_res |= tag(run_intent({"llm_zero_shot": LLMIntent(llm)}, sub, op(sub_conds)), "[subset]")
        ext_res |= tag(run_extraction(sub, sub_conds, {"llm": lambda turns, d: llm_extract(llm, turns, d)}), "[subset]")
        u = llm.usage
        usage = {
            "calls": u.calls,
            "prompt_tokens": u.prompt_tokens,
            "completion_tokens": u.completion_tokens,
            "seconds": u.seconds,
            "failures": u.failures,
            "repairs": u.repairs,
            "llm_subset_n": len(sub),
        }
    dev_sel = _items("dev")
    sel_cond = "asr" if "asr" in conds and all("text_asr" in it for it in dev_sel) else "reference"
    selective = run_selective(tfidf, dev_sel, test, sel_cond)
    _write(
        "nlp",
        {
            "test_calls": len(test),
            "conditions": conds,
            "intent": intent_res,
            "extraction": ext_res,
            "selective": selective,
            "llm_usage": usage,
        },
    )


def cmd_eval_rules(a) -> None:
    """Re-score the current rules version without the LLM. Written to a separate file: rules
    changes made after the frozen test run are post-hoc and reported as such."""
    from cci.eval.nlp_eval import run_extraction, run_selective
    from cci.nlp.intent import TfidfIntent
    from cci.nlp.rules import RULES_VERSION
    from cci.nlp.rules import extract as rule_extract

    train, dev, test = _items("train"), _items("dev"), _items("test")
    tfidf = TfidfIntent().fit([it["text_reference"] for it in train], [it["label"]["intent"] for it in train])
    res = {"rules_version": RULES_VERSION, "post_hoc": True}
    for name, items in (("dev", dev), ("test", test)):
        res[name] = run_extraction(items, ["reference", "asr"], {"rules": rule_extract})
    sub = test[:60]
    res["test_subset"] = run_extraction(sub, ["reference", "asr", "paraphrase"], {"rules": rule_extract})
    res["selective"] = run_selective(tfidf, dev, test, "asr")
    _write(f"rules_v{RULES_VERSION.replace('.', '_')}", res)


def cmd_eval_rag(a) -> None:
    from cci.eval.rag_eval import eval_rag

    _write(
        "rag",
        eval_rag(DATA / "kb", DATA / "kb_qa.jsonl", None if a.no_llm else _llm(), None if a.no_embed else _embedder()),
    )


def cmd_train_intent(a) -> None:
    from cci.training import train_and_register

    res = train_and_register(
        _items("train"), _items("dev"), os.environ.get("CCI_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    print(json.dumps(res, indent=1))
    # Also export the champion as a plain artifact for deployments without an MLflow server.
    from cci.app.models import load_intent_model
    from cci.training import save_local

    model, version = load_intent_model(
        "models:/cci-intent@champion", os.environ.get("CCI_MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
    )
    save_local(model, MODELS / "intent" / "intent.joblib")
    import joblib

    from cci.speech.diarization import RoleModel

    train = _calls("train")
    roles = RoleModel().fit(
        [t["text"] for c in train for t in c["turns"]], [t["speaker"] for c in train for t in c["turns"]]
    )
    joblib.dump(roles, MODELS / "intent" / "roles.joblib")
    print(f"exported {version} to {MODELS / 'intent' / 'intent.joblib'}")


def cmd_serve(a) -> None:
    import uvicorn

    from cci.app.api import create_app
    from cci.app.settings import Settings

    s = Settings()
    answerer = None
    if s.llm_base_url:
        from cci.llm.client import LLMClient, LLMConfig
        from cci.rag.answer import answer_grounded
        from cci.rag.retrieval import HybridIndex, load_kb

        llm = LLMClient(LLMConfig(base_url=s.llm_base_url))
        embed = LLMClient(LLMConfig(base_url=s.embed_base_url)).embed if s.embed_base_url else None
        index = HybridIndex(load_kb(s.kb_dir), embed)
        answerer = lambda q: answer_grounded(llm, index, q)  # noqa: E731
    uvicorn.run(create_app(s, answerer), host=a.host, port=a.port)


def cmd_worker(a) -> None:
    from cci.app.worker import main as worker_main

    worker_main()


def cmd_report(a) -> None:
    from cci.eval.report import build_report

    print(f"wrote {build_report(RESULTS)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="cci")
    sub = p.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("corpus")
    c.add_argument("--train", type=int, default=600)
    c.add_argument("--dev", type=int, default=150)
    c.add_argument("--test", type=int, default=150)
    c.add_argument("--seed", type=int, default=20261022)
    c.set_defaults(fn=cmd_corpus)
    for name, fn in (
        ("render", cmd_render),
        ("transcribe", cmd_transcribe),
        ("eval-asr", cmd_eval_asr),
        ("paraphrase", cmd_paraphrase),
    ):
        s = sub.add_parser(name)
        s.add_argument("--split", default="test")
        s.add_argument("--condition", default="telephony", choices=list(CONDITIONS))
        s.add_argument("--limit", type=int, default=None)
        s.add_argument("--threads", type=int, default=4)
        if name == "eval-asr":
            s.add_argument("--backends", default="gigaam-v2,parakeet-v3,whisper-small")
            s.add_argument("--diarization", action="store_true")
        s.set_defaults(fn=fn)
    n = sub.add_parser("eval-nlp")
    n.add_argument("--limit", type=int, default=None)
    n.add_argument("--no-llm", action="store_true")
    n.add_argument("--llm-limit", type=int, default=60)
    n.set_defaults(fn=cmd_eval_nlp)
    sub.add_parser("eval-rules").set_defaults(fn=cmd_eval_rules)
    r = sub.add_parser("eval-rag")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--no-embed", action="store_true")
    r.set_defaults(fn=cmd_eval_rag)
    ro = sub.add_parser("roles")
    ro.add_argument("--split", default="dev")
    ro.add_argument("--condition", default="telephony")
    ro.add_argument("--apply", action="store_true")
    ro.add_argument("--mode", default="text", choices=["text", "cluster", "hybrid"])
    ro.set_defaults(fn=cmd_roles)
    sub.add_parser("train-intent").set_defaults(fn=cmd_train_intent)
    sv = sub.add_parser("serve")
    sv.add_argument("--host", default="127.0.0.1")
    sv.add_argument("--port", type=int, default=8090)
    sv.set_defaults(fn=cmd_serve)
    sub.add_parser("worker").set_defaults(fn=cmd_worker)
    sub.add_parser("report").set_defaults(fn=cmd_report)
    a = p.parse_args(argv)
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
