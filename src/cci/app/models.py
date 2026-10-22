"""Loading the intent model from the MLflow Model Registry (or a local artifact)."""

from __future__ import annotations

from pathlib import Path


def load_intent_model(uri: str, tracking_uri: str):
    """``models:/name@alias`` -> MLflow registry; anything else -> local joblib file."""
    if uri.startswith("models:/"):
        import mlflow

        mlflow.set_tracking_uri(tracking_uri)
        model = mlflow.sklearn.load_model(uri)
        version = _registry_version(uri, tracking_uri)
    else:
        import joblib

        model = joblib.load(Path(uri))
        version = Path(uri).name
    from cci.nlp.intent import TfidfIntent

    wrapper = TfidfIntent.__new__(TfidfIntent)
    wrapper.model = model
    return wrapper, version


def _registry_version(uri: str, tracking_uri: str) -> str:
    from mlflow import MlflowClient

    name, _, alias = uri.removeprefix("models:/").partition("@")
    if not alias:
        return uri
    mv = MlflowClient(tracking_uri=tracking_uri).get_model_version_by_alias(name, alias)
    return f"{name}/{mv.version}"
