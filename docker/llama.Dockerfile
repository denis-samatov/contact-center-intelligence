# llama.cpp server built from source (pinned by commit for reproducibility).
FROM mirror.gcr.io/library/debian:bookworm-slim AS build
ARG LLAMA_COMMIT=5f5acfe2504b490386cd03020202f59e36861add
RUN apt-get update && apt-get install -y --no-install-recommends git cmake build-essential ca-certificates libgomp1 \
    && git clone https://github.com/ggml-org/llama.cpp /src && cd /src && git checkout ${LLAMA_COMMIT} \
    && cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    && cmake --build build --target llama-server -j
FROM mirror.gcr.io/library/debian:bookworm-slim
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY --from=build /src/build/bin/ /opt/llama/
ENV LD_LIBRARY_PATH=/opt/llama
ENTRYPOINT ["/opt/llama/llama-server"]
