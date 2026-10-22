# ADR-B5: Local-first LLM behind an OpenAI-compatible API

Status: accepted (2026-10-22)

## Decision
LLM features (zero-shot intent, LLM extraction, grounded answers, paraphrase generation)
call an OpenAI-compatible endpoint. Default: `llama-server` (llama.cpp built from source)
serving Qwen3-1.7B Q4_K_M and Qwen3-Embedding-0.6B Q8_0 on CPU. vLLM/SGLang or an external
provider can be configured with `CCI_LLM_BASE_URL` / `CCI_LLM_API_KEY`; nothing external is
called by default, and no paid API was used for any result.

## Notes
* Weights came from Docker Hub model artifacts (`ai/qwen3`, `ai/qwen3-embedding`, OCI/CNCF
  model spec); SHA-256 of downloaded blobs equals the manifest digests.
* Structured output: JSON schema in `response_format`, then Pydantic validation, one repair
  attempt, then failure -> human review.
* The rule-based extractor remains the default in the worker: it is faster, deterministic,
  and on this corpus at least as accurate (see results); the LLM path is evaluated, not
  assumed better.
