# syntax=docker/dockerfile:1
# CPU image for the API and the worker (same image, different command).
# libsndfile comes with the soundfile wheel, so no system packages are needed.
FROM mirror.gcr.io/library/python:3.12-slim
# The storage dir must exist with the right owner: a fresh named volume copies it (else root-owned).
RUN useradd -u 10001 -m app && mkdir -p /data/storage && chown app /data/storage
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data/kb ./data/kb
COPY results ./results
# Optional: a corporate CA for TLS-intercepting proxies (docker build --secret id=pip_ca,src=ca.pem).
RUN --mount=type=secret,id=pip_ca,required=false \
    if [ -f /run/secrets/pip_ca ]; then export PIP_CERT=/run/secrets/pip_ca; fi; \
    pip install --no-cache-dir .
USER 10001
ENV CCI_MODELS_DIR=/models CCI_STORAGE_DIR=/data/storage CCI_KB_DIR=/app/data/kb CCI_RESULTS_DIR=/app/results
EXPOSE 8090
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/health/live')"
CMD ["cci", "serve", "--host", "0.0.0.0", "--port", "8090"]
