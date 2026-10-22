# Evaluation protocol

Fixed before the reported runs (corpus seed 20261023).

## Data
* 600 train / 150 dev / 150 test synthetic calls (`cci corpus`), 9 intents, gold fields,
  contradictions, missing required fields, explicit preferences, queue and next actions.
* Client openings come from template families: train {A, B}, dev {C}, test {D}. Family D
  phrasings are deliberately indirect ("my car is in the service after what happened").
* Audio: CC0 Piper voices (operator `dmitri`, client `denis`), 16 kHz. Conditions:
  `clean`; `telephony` (8 kHz band-limit + mu-law); `noisy` (telephony + noise at 10 dB SNR).
* Paraphrase set: client turns of the first 60 test calls rewritten by Qwen3-1.7B; a rewrite
  is kept only if numbers (parsed), named places/streets and key cue words are preserved.

## Tasks and metrics
| task | input | metric | notes |
|---|---|---|---|
| ASR | audio | corpus WER (95% bootstrap CI over calls), CER, RTF | numbers normalized to digits on both sides; raw WER also reported |
| Diarization | audio | DER (no collar), operator/client role accuracy | 2 speakers, both male CC0 voices |
| Intent (post-call) | full transcript | accuracy, macro-F1 + CIs | trivial on this corpus: operator questions are intent-specific |
| Intent (early routing) | first client utterance | accuracy, macro-F1 + CIs | the meaningful comparison |
| Extraction | role-labelled turns | micro P/R/F1 over (call, field) of the gold intent's schema | contradicted fields excluded from value scoring |
| Contradictions | turns | set P/R | |
| Preferences | turns | set F1 | only explicit statements |
| Selective automation | ASR condition, full pipeline | automation rate, error rate on automated cases | threshold chosen on dev for <= 2% case error |
| KB answers | 40 questions over 30 chunks | retrieval recall@k; fact recall closed-book vs RAG; verified-grounded rate; citation precision | fictional rulebook |

## Comparisons
* Classical (TF-IDF + LR, rules) vs neural/LLM (Qwen3-Embedding + LR, Qwen3-1.7B zero-shot,
  LLM JSON extraction). LLM methods run on the first 60 test calls (CPU cost); classical
  methods are reported on the same 60 (`[subset]` rows) *and* on all 150.
* ASR: GigaAM-v2 vs Parakeet-TDT-0.6B-v3 vs Whisper-small on the same 60 test calls.
* No retrieval vs grounded generation on the same questions and model.
* Automate-all vs review-on-missing/contradiction vs + confidence threshold.

## What this can and cannot show
It compares methods under identical, controlled conditions and validates the pipeline end to
end. It cannot estimate accuracy on real customer calls (spontaneous speech, overlaps,
accents, real noise), and the test phrasings, although disjoint from training, were written by
the same author as the generator.
