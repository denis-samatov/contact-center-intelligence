# Reproducible local CPU/dev profile

`requirements-cpu-py313-macos.lock` was resolved from `pyproject.toml --extra dev`
for Python3.13 / Apple Silicon macOS during audit remediation. It records
registry hashes and, where used, immutable VCS commit IDs. It is not a complete
artifact signature or a claim about Linux/CUDA/production operation.

```sh
uv venv --python 3.13 .venv
uv pip sync --python .venv/bin/python requirements-cpu-py313-macos.lock
uv pip install --python .venv/bin/python --no-deps -e .
PYTHONPATH=src .venv/bin/python -m pytest -q
```

Regenerate with the command in the lock header. Do not hand-edit the generated
lock, override a conflicting upstream version, or substitute a mutable Git HEAD.
Hosted-service/model/data integrations remain opt-in; ordinary synthetic unit
checks do not establish real-domain quality, privacy acceptance or deployment.
The profile does not freeze remote model aliases, ONNX weights, corpora or
PostgreSQL server state; record those hashes/versions in each actual qualification.
