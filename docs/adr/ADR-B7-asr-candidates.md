# ADR-B7: ASR candidates

Status: accepted (2026-10-22)

| candidate | license | why |
|---|---|---|
| GigaAM-v2 RNN-T (ONNX int8) | MIT (upstream salute-developers/GigaAM; the LICENSE file inside the sherpa-onnx archive is empty) | Russian-specific; strong reported Russian WER; token timestamps |
| Parakeet-TDT-0.6B-v3 (ONNX int8) | CC-BY-4.0 | multilingual SOTA-class transducer incl. Russian |
| Whisper small (ONNX) | MIT | widely used baseline; seq2seq, writes digits |
| Qwen3-ASR-0.6B/1.7B | Apache-2.0 | **not evaluated**: weights only on Hugging Face/ModelScope, both blocked here |

Decision: GigaAM-v2 is the default if it wins on the dev/test audio at acceptable RTF; the
choice is re-checked by the evaluation, not assumed.
