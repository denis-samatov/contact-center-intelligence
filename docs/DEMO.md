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

## Video walkthrough script (about 3 minutes)

| time | on screen | narration |
|---|---|---|
| 0:00-0:15 | README header, "Data" row | "From a recorded insurance call to a verified case card. All calls are synthetic: generated dialogues voiced with CC0 voices, so no real customer data is involved." |
| 0:15-0:40 | UI: upload `test-0013.wav`, provenance `synthetic`; worker log `job_done` | "Uploads must declare provenance. The job goes through a PostgreSQL queue; the worker transcribes 48 seconds of telephone audio in about 16 seconds on CPU." |
| 0:40-1:20 | case card (`docs/img/case.png` state) | "GigaAM transcript with timestamps and roles, the complaint fields, and a contradiction: the client gave two different policy numbers. The case goes to review with that reason, and the operator can correct any field." |
| 1:20-1:45 | `results/REPORT.md`, ASR table | "Three ASR models on the same 60 calls: GigaAM-v2 9.3% WER on telephone audio, Parakeet 16.4%, Whisper-small 31.6%." |
| 1:45-2:20 | automation table | "Automating every case would be wrong 38% of the time. Sending incomplete or contradictory cases to review automates 55% at 8.4% error. An intent-confidence threshold could not help: the errors are in fields, not in intent." |
| 2:20-2:45 | KB answer in the UI, RAG table | "Answers about the rulebook come with citations and a numeric check against the cited text: 80% fully correct with retrieval versus 5% without, same model." |
| 2:45-3:00 | `docs/EVALUATION.md` post-hoc section | "One honest failure: the frozen rules missed every contradiction in ASR text because of a hyphen. The fix was developed on dev and its test numbers are labelled post-hoc." |
