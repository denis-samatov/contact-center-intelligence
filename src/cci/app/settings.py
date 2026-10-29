"""Validated runtime configuration (environment variables prefixed ``CCI_``)."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict

ALLOWED_PROVENANCE = ("synthetic", "licensed", "consented_recording")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CCI_", extra="ignore")

    database_url: str = "sqlite:///./cci.db"
    storage_dir: Path = Path("./storage")
    models_dir: Path = Path("./models")
    kb_dir: Path = Path("./data/kb")
    results_dir: Path = Path("./results")  # committed offline evaluation shown on the dashboard
    api_key: SecretStr | None = Field(None, min_length=1)  # required unless allow_anonymous_local
    allow_anonymous_local: bool = False
    max_upload_mb: int = Field(50, ge=1, le=500)
    rate_limit_per_min: int = Field(120, ge=1)  # per API key (or client address without a key)
    asr_backend: str = "gigaam-v2"
    diarize: bool = True
    num_threads: int = Field(4, ge=1)
    intent_model_uri: str = "models:/cci-intent@champion"  # MLflow registry URI, or a local .joblib path
    mlflow_tracking_uri: str = "sqlite:///./mlflow.db"
    llm_base_url: str | None = None  # e.g. http://127.0.0.1:8081/v1 (llama.cpp). None disables LLM features
    embed_base_url: str | None = None
    min_intent_prob: float = Field(0.6, ge=0, le=1)
    worker_poll_s: float = 1.0
    job_visibility_timeout_s: int = 900
