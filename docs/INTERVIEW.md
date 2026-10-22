# Technical defense notes

**Q: Your data is synthetic. Why should I believe any number?**
Believe the comparisons, not the absolute levels. All methods see the same audio and the
same gold labels; test phrasings come from a template family never used in training; a third
condition rewrites client turns with an LLM under fact checks. Where a result is trivially
high by construction (rules on reference text, intent on full transcripts) it is labelled so.
A real pilot on consented recordings is the next step (docs/BUSINESS_CASE.md).

**Q: Why classify intent on the first utterance?**
On full transcripts the operator's questions reveal the intent (each intent asks for its own
fields): TF-IDF reaches 100%, which says nothing. Early routing from the first client
utterance is both the harder and the operationally useful task.

**Q: Why is the rule extractor the default and not the LLM?**
It is deterministic, fast and auditable, and the evaluation decides: if the LLM is not more
accurate under ASR noise and paraphrase, it does not justify its latency on CPU. The LLM
path stays available and evaluated (ADR-B5).

**Q: How does selective automation work?**
A case goes to a human if a required field is missing, if the customer contradicted
themselves, or if intent probability is below a threshold chosen on dev. Measured on ASR text:
automate-all is wrong on 38% of cases; review-on-missing/contradiction automates 55% with 8.4%
errors. No intent threshold reached 2% on dev, because the remaining errors are in fields, not
intent - so the threshold sends everything to review. That is the finding: the next signal
has to be per-field confidence, not a better intent model.

**Q: How independent are your test calls?**
Less than 150 suggests. Integrating localdesk surfaced it: its leakage guard refused a split
because identical openings crossed it. Test has 27 distinct first utterances for 150 calls, so I
added template-cluster bootstrap intervals; they are 2-2.5x wider. Embeddings still lead
TF-IDF (0.80 vs 0.58 on ASR openings) but the intervals overlap, so I do not call it established.

**Q: What went wrong during the evaluation?**
The frozen rules scored contradiction F1 = 0 on ASR: the operator cue "что-нибудь еще" was
matched literally and ASR drops the hyphen. I reproduced it on dev, fixed it there (v1.1) and
report the test numbers as post-hoc next to the frozen ones. The first paraphrase run kept 7%
of turns; with inlined few-shot examples the 1.7B model copied the example, and one copy passed
the fact checks - so a copy check and an embedding gate were added (debugged on dev).

**Q: How do you prevent hallucinated answers?**
Answers are generated from retrieved rule chunks with citations, then verified: cited ids
must be among retrieved chunks, and every number in the answer must occur in the cited
text. Unverified answers are shown as "needs review". The closed-book baseline on the
fictional rulebook measures how often a model invents specifics.

**Q: What about personal data and ethics?**
Uploads require declared provenance (synthetic, licensed, consented recording); no real calls
are used; routing uses only stated intent and explicit preferences; no emotion, demographic or
health inference; no automated insurance decisions (ADR-B4). TTS voices were chosen by their
dataset licenses (ADR-B3).

**Q: Why PostgreSQL as a queue and not Kafka/Redis?**
Job state and case state commit in one transaction; `SKIP LOCKED` gives safe concurrent
workers; volumes of a contact center fit easily (ADR-B2). Airflow is used only for the nightly
periodic work, where retries and backfills matter.
