#!/usr/bin/env bash
# Download the ONNX speech models (sherpa-onnx GitHub release assets) and LLM GGUFs.
# Licenses: see docs/adr/ADR-B3 and ADR-B7. Only CC0 Piper voices are used for audio.
set -euo pipefail
DEST=${1:-models}
mkdir -p "$DEST" && cd "$DEST"
B=https://github.com/k2-fsa/sherpa-onnx/releases/download
for a in asr-models/sherpa-onnx-nemo-transducer-giga-am-v2-russian-2025-04-19.tar.bz2 \
         asr-models/sherpa-onnx-nemo-parakeet-tdt-0.6b-v3-int8.tar.bz2 \
         asr-models/sherpa-onnx-whisper-small.tar.bz2 asr-models/silero_vad.onnx \
         tts-models/vits-piper-ru_RU-denis-medium.tar.bz2 tts-models/vits-piper-ru_RU-dmitri-medium.tar.bz2 \
         speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2 \
         speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx; do
  f=$(basename "$a")
  [ -s "$f" ] || curl -fL --retry 4 -o "$f" "$B/$a"
  case "$f" in *.tar.bz2) d="${f%.tar.bz2}"; [ -d "$d" ] || tar xjf "$f";; esac
done
# LLM weights: Docker Hub model artifacts (OCI). Any OpenAI-compatible server works instead.
echo "LLM: pull ai/qwen3:1.7b-q4_K_M and ai/qwen3-embedding:0.6b-q8_0 (docker model pull) or point CCI_LLM_BASE_URL elsewhere."
