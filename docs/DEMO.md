# Demo script (about 4 minutes)

Setup: `scripts/fetch_models.sh models`, llama-server for chat (8081) and embeddings (8082),
`cci corpus && cci render --split test --condition telephony --limit 5 && cci train-intent`,
then `CCI_API_KEY=demo CCI_INTENT_MODEL_URI=models/intent/intent.joblib cci serve` and `cci worker`.

1. **Problem (20 s)**: after-call work in an insurance contact center; show a raw recording.
2. **Upload (30 s)**: upload `data/audio/test/telephony/test-0003.wav`, provenance `synthetic`.
   Try provenance "scraped" -> rejected (422).
3. **Transcript (40 s)**: timestamps, operator/client roles from diarization, numbers
   normalized ("ноль два ..." -> digits).
4. **Case card (60 s)**: intent with probability, fields, missing fields in red, a
   contradiction highlighted, queue, next actions, decision `review` with reasons.
5. **Correction (30 s)**: fix a field, approve; version increments; second edit with a stale
   version is rejected (409) - show in the API.
6. **Grounded answer (30 s)**: ask "Какая франшиза по КАСКО?"; show citation and the
   "подтвержден источниками" check.
7. **Evidence (30 s)**: `results/REPORT.md`: ASR candidates, rules vs LLM under ASR and
   paraphrase, automate-all vs selective.
