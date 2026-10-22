"""Train, evaluate and register the intent classifier with MLflow (tracking + Model Registry).

Promotion rule (quality gate): a new version receives the ``champion`` alias only if its
dev macro-F1 is not lower than the current champion's by more than ``tolerance``. The
previous champion stays registered, so rollback is re-pointing the alias.
"""

from __future__ import annotations

import json
from pathlib import Path

from cci.eval.metrics import macro_f1
from cci.nlp.intent import TfidfIntent

MODEL_NAME = "cci-intent"


def _texts(items: list[dict]) -> tuple[list[str], list[str]]:
    return [it["text_reference"] for it in items], [it["label"]["intent"] for it in items]


def train_and_register(
    train: list[dict],
    dev: list[dict],
    tracking_uri: str,
    extra_train: list[dict] | None = None,
    tolerance: float = 0.01,
    run_name: str = "tfidf-intent",
) -> dict:
    import mlflow
    from mlflow import MlflowClient

    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment("cci-intent")
    X, y = _texts(train + (extra_train or []))
    Xd, yd = _texts(dev)
    with mlflow.start_run(run_name=run_name) as run:
        model = TfidfIntent().fit(X, y)
        pred, _ = model.predict_proba(Xd)
        f1 = macro_f1(yd, pred)
        acc = sum(a == b for a, b in zip(yd, pred, strict=True)) / len(yd)
        mlflow.log_params(
            {"model": "tfidf_word12_char25_lr", "C": 10, "n_train": len(X), "n_feedback": len(extra_train or [])}
        )
        mlflow.log_metrics({"dev_macro_f1": f1, "dev_accuracy": acc})
        info = mlflow.sklearn.log_model(
            model.model, name="model", registered_model_name=MODEL_NAME, input_example=[X[0]]
        )
        client = MlflowClient(tracking_uri=tracking_uri)
        version = info.registered_model_version
        try:
            champ = client.get_model_version_by_alias(MODEL_NAME, "champion")
            champ_f1 = client.get_run(champ.run_id).data.metrics.get("dev_macro_f1", 0.0)
        except Exception:  # noqa: BLE001 - no champion yet
            champ, champ_f1 = None, None
        promoted = champ_f1 is None or f1 >= champ_f1 - tolerance
        if promoted:
            client.set_registered_model_alias(MODEL_NAME, "champion", version)
        client.set_model_version_tag(MODEL_NAME, version, "gate", "passed" if promoted else "failed")
        result = {
            "run_id": run.info.run_id,
            "version": version,
            "dev_macro_f1": f1,
            "dev_accuracy": acc,
            "champion_before": None if champ is None else champ.version,
            "champion_dev_macro_f1": champ_f1,
            "promoted": promoted,
        }
        mlflow.log_dict(result, "promotion.json")
        return result


def feedback_items(reviews: list[dict]) -> list[dict]:
    """Turn reviewed cases (human-corrected intent) into extra training items."""
    return [{"text_reference": r["text"], "label": {"intent": r["intent"]}} for r in reviews if r.get("text")]


def save_local(model: TfidfIntent, path: Path) -> None:
    import joblib

    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model.model, path)
    path.with_suffix(".json").write_text(json.dumps({"format": "sklearn-pipeline-joblib"}))
