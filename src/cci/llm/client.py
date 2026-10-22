"""Provider-agnostic LLM client for OpenAI-compatible chat/embedding endpoints.

The same code path talks to a local ``llama-server`` (llama.cpp, the default and the only
provider exercised in this repository's measurements), to vLLM / SGLang servers, or to an
explicitly configured external provider. Nothing external is called unless
``CCI_LLM_BASE_URL`` / ``CCI_LLM_API_KEY`` are set by the operator.

Structured outputs: requests carry a JSON schema (``response_format``); the reply is then
validated with Pydantic. A schema violation triggers one repair attempt that includes the
validation error, then the call fails loudly - callers route such cases to human review.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    pass


@dataclass
class Usage:
    calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    seconds: float = 0.0
    failures: int = 0
    repairs: int = 0
    latencies: list[float] = field(default_factory=list)

    def add(self, other: dict, seconds: float) -> None:
        self.calls += 1
        self.prompt_tokens += int(other.get("prompt_tokens", 0))
        self.completion_tokens += int(other.get("completion_tokens", 0))
        self.seconds += seconds
        self.latencies.append(seconds)


@dataclass(frozen=True)
class LLMConfig:
    base_url: str = "http://127.0.0.1:8081/v1"
    model: str = "qwen3-1.7b-q4_k_m"
    api_key: str | None = None
    timeout_s: float = 180.0
    max_retries: int = 2
    temperature: float = 0.0
    max_tokens: int = 512

    @classmethod
    def from_env(cls, prefix: str = "CCI_LLM_") -> LLMConfig:
        env = os.environ
        return cls(
            base_url=env.get(f"{prefix}BASE_URL", cls.base_url),
            model=env.get(f"{prefix}MODEL", cls.model),
            api_key=env.get(f"{prefix}API_KEY") or None,
            timeout_s=float(env.get(f"{prefix}TIMEOUT_S", cls.timeout_s)),
        )


class LLMClient:
    def __init__(self, cfg: LLMConfig | None = None, transport: httpx.BaseTransport | None = None):
        self.cfg = cfg or LLMConfig.from_env()
        headers = {"Authorization": f"Bearer {self.cfg.api_key}"} if self.cfg.api_key else {}
        self._http = httpx.Client(
            base_url=self.cfg.base_url, timeout=self.cfg.timeout_s, headers=headers, transport=transport
        )
        self.usage = Usage()

    def _post(self, path: str, payload: dict) -> dict:
        last: Exception | None = None
        for attempt in range(self.cfg.max_retries + 1):
            t0 = time.perf_counter()
            try:
                r = self._http.post(path, json=payload)
                if r.status_code >= 500 or r.status_code == 429:
                    raise httpx.HTTPStatusError(f"{r.status_code}", request=r.request, response=r)
                r.raise_for_status()
                data = r.json()
                self.usage.add(data.get("usage", {}), time.perf_counter() - t0)
                return data
            except (httpx.TransportError, httpx.HTTPStatusError) as e:
                last = e
                time.sleep(0.5 * 2**attempt)
        self.usage.failures += 1
        raise LLMError(f"LLM request failed after retries: {last}")

    def chat(
        self,
        messages: list[dict],
        schema: dict | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
    ) -> str:
        payload = {
            "model": self.cfg.model,
            "messages": messages,
            "temperature": self.cfg.temperature if temperature is None else temperature,
            "max_tokens": max_tokens or self.cfg.max_tokens,
        }
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": "output", "schema": schema},
                "schema": schema,
            }
        data = self._post("/chat/completions", payload)
        return data["choices"][0]["message"].get("content") or ""

    def structured(
        self, messages: list[dict], model: type[T], max_tokens: int | None = None, temperature: float | None = None
    ) -> T:
        schema = model.model_json_schema()
        text = self.chat(messages, schema=schema, max_tokens=max_tokens, temperature=temperature)
        try:
            return model.model_validate_json(_strip(text))
        except ValidationError as e:
            self.usage.repairs += 1
            repair = messages + [
                {"role": "assistant", "content": text},
                {"role": "user", "content": f"The JSON is invalid: {e.errors()[:3]}. Return only corrected JSON."},
            ]
            text = self.chat(repair, schema=schema, max_tokens=max_tokens)
            try:
                return model.model_validate_json(_strip(text))
            except ValidationError as e2:
                raise LLMError(f"schema validation failed twice: {e2.errors()[:3]}") from e2

    def embed(self, texts: list[str], batch: int = 16) -> list[list[float]]:
        out: list[list[float]] = []
        for i in range(0, len(texts), batch):
            data = self._post("/embeddings", {"model": self.cfg.model, "input": texts[i : i + batch]})
            out += [d["embedding"] for d in sorted(data["data"], key=lambda d: d["index"])]
        return out

    def health(self) -> bool:
        try:
            r = httpx.get(self.cfg.base_url.removesuffix("/v1") + "/health", timeout=5)
            return r.status_code == 200
        except httpx.HTTPError:
            return False


def _strip(text: str) -> str:
    """Drop think blocks / code fences some models emit around JSON."""
    text = text.strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[1].strip()
    if text.startswith("```"):
        text = text.strip("`").removeprefix("json").strip()
    return text


def dumps(obj) -> str:
    return json.dumps(obj, ensure_ascii=False)
