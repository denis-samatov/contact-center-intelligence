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
A case goes to a human if intent probability is below a threshold chosen on dev, if a
required field is missing, or if the customer contradicted themselves. The report shows
automation rate and error rate *on automated cases* for automate-all vs selective policies.

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
