# ADR-B8: Operator/client roles from a text model, not from voice clusters

Status: accepted (2026-10-22). Selected on dev; applied once to test.

| method (telephony audio, GigaAM transcripts) | dev role accuracy | dev calls < 0.9 | test role accuracy | test calls < 0.9 |
|---|---:|---:|---:|---:|
| script-cue heuristic over diarization clusters | 0.896 | 31 | 0.896 | 26 |
| text model, majority per cluster | 0.953 | 26 | - | - |
| text model, hybrid (cluster unless confident) | 0.995 | 4 | - | - |
| **text model per segment** (chosen) | **0.997** | **0** | **0.9985** | **0** |

The text model is TF-IDF + logistic regression trained on train-split reference turns.
Failure modes of the cluster-based methods: with two similar male voices the clustering
sometimes splits a speaker or collapses to one cluster, and a whole call's roles flip.

Caveat: operators in this corpus follow a fixed script, which makes text a strong signal by
construction. Real contact centers also script operators, but less strictly; the production
choice should be re-validated on real calls, keeping diarization as a second signal (hybrid).
