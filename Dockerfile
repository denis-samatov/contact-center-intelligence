# CPU image for the API and the worker (same image, different command).
FROM mirror.gcr.io/library/python:3.12-slim
RUN apt-get update && apt-get install -y --no-install-recommends libsndfile1 && rm -rf /var/lib/apt/lists/* \
    && useradd -u 10001 -m app
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
COPY data/kb ./data/kb
RUN pip install --no-cache-dir .
USER 10001
ENV CCI_MODELS_DIR=/models CCI_STORAGE_DIR=/data/storage CCI_KB_DIR=/app/data/kb
EXPOSE 8090
HEALTHCHECK CMD python -c "import urllib.request;urllib.request.urlopen('http://127.0.0.1:8090/health/live')"
CMD ["cci", "serve", "--host", "0.0.0.0", "--port", "8090"]
