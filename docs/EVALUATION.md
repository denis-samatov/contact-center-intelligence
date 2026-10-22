# Evaluation protocol

Fixed before the reported runs (corpus seed 20261023).

## Data
* 600 train / 150 dev / 150 test synthetic calls (`cci corpus`), 9 intents, gold fields,
  contradictions, missing required fields, explicit preferences, queue and next actions.
* Client openings come from template families: train {A, B}, dev {C}, test {D}. Family D
  phrasings are deliberately indirect ("my car is in the service after what happened").
* Audio: CC0 Piper voices (operator `dmitri`, client `denis`), 16 kHz. Conditions:
  `clean`; `telephony` (8 kHz band-limit + mu-law); `noisy` (telephony + noise at 10 dB SNR).
* Paraphrase set: client turns of the first 60 test calls rewritten by Qwen3-1.7B
  (temperature 0.7, two few-shot pairs as separate chat turns); a rewrite is kept only if
  numbers (parsed), named places/streets and key cue words are preserved, it differs from
  the input, it does not copy a few-shot answer, and its embedding cosine to the original is
  >= 0.6 (garbage gate, below the 5th percentile seen on dev). The prompt was debugged on dev
  calls only: the first version (instruction-only, temperature 0) kept 7% of test turns,
  251 of 263 rejections being "unchanged or too long" (not broken down at the time); with examples inlined in the prompt the model copied
  the example verbatim, and one such copy passed the fact checks - hence the copy check.
  Rejected turns keep the original text; acceptance statistics are in `results/paraphrase_test.json`.

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

## Independence of observations
Calls generated from the same template are not independent. The test split has 27 distinct
first utterances for 150 calls (dev 27, train 54). Early-routing intent is therefore reported
with template-cluster bootstrap CIs (`cci eval-intent`, `results/intent_openings.json`); the
call-level intervals of `results/nlp.json` are kept for continuity but are too narrow. Field
extraction is less affected (values such as numbers, dates and addresses are drawn per call),
but phrasings still repeat, so its intervals should also be read as optimistic.

## What this can and cannot show
It compares methods under identical, controlled conditions and validates the pipeline end to
end. It cannot estimate accuracy on real customer calls (spontaneous speech, overlaps,
accents, real noise), and the test phrasings, although disjoint from training, were written by
the same author as the generator.
