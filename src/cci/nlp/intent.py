"""Intent classifiers compared in the evaluation.

* :class:`TfidfIntent` - classical baseline: word 1-2 + char 2-5 TF-IDF, logistic regression.
  Char n-grams make it tolerant to ASR misspellings.
* :class:`EmbeddingIntent` - neural sentence embeddings (Qwen3-Embedding-0.6B via an
  OpenAI-compatible endpoint) + logistic regression.
* :class:`LLMIntent` - zero-shot instruction-following LLM with a JSON-schema constrained
  label (no training data). It returns a label only; LLM self-reported confidence is not
  used for routing decisions because it is not calibrated.

All classifiers expose ``predict_proba(texts) -> (labels, probs[n, k])`` so that the
selective-prediction layer can treat them uniformly (the LLM returns one-hot rows).
"""

from __future__ import annotations

from enum import Enum

import numpy as np
from pydantic import BaseModel, create_model

from cci.domain.schema import Intent

DESCRIPTIONS = {
    Intent.CLAIM_AUTO: "заявление о страховом случае с автомобилем: ДТП, авария, повреждение машины",
    Intent.CLAIM_PROPERTY: "заявление о страховом случае с квартирой или имуществом: залив, пожар, кража",
    Intent.PAYMENT_STATUS: "вопрос о статусе или сроках выплаты по уже поданному заявлению",
    Intent.POLICY_RENEWAL: "продление полиса на новый срок",
    Intent.POLICY_CANCELLATION: "расторжение договора, отказ от полиса, возврат премии",
    Intent.DATA_CHANGE: "изменение данных в договоре: адрес, телефон, водители",
    Intent.COMPLAINT: "жалоба или претензия: недовольство сроками, суммой или обслуживанием",
    Intent.DOCUMENT_REQUEST: "запрос документа: копия полиса, справка о выплате, решение по убытку",
    Intent.GENERAL_INFO: "общий вопрос об условиях страхования, франшизе, сроках, документах",
}
LABELS = [i.value for i in Intent]


class TfidfIntent:
    name = "tfidf_lr"

    def __init__(self, C: float = 10.0):
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import FeatureUnion, make_pipeline

        self.model = make_pipeline(
            FeatureUnion(
                [
                    ("w", TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)),
                    ("c", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), sublinear_tf=True)),
                ]
            ),
            LogisticRegression(C=C, max_iter=3000),
        )

    def fit(self, texts: list[str], labels: list[str]) -> TfidfIntent:
        self.model.fit(texts, labels)
        return self

    def predict_proba(self, texts: list[str]) -> tuple[list[str], np.ndarray]:
        p = self.model.predict_proba(texts)
        classes = list(self.model.classes_)
        full = np.zeros((len(texts), len(LABELS)))
        for j, c in enumerate(classes):
            full[:, LABELS.index(c)] = p[:, j]
        return [LABELS[i] for i in full.argmax(1)], full


class EmbeddingIntent:
    name = "embedding_lr"

    def __init__(self, embed, C: float = 10.0):
        from sklearn.linear_model import LogisticRegression

        self.embed = embed
        self.clf = LogisticRegression(C=C, max_iter=3000)

    def _vec(self, texts):
        v = np.asarray(self.embed(texts), dtype=np.float32)
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def fit(self, texts, labels):
        self.clf.fit(self._vec(texts), labels)
        return self

    def predict_proba(self, texts):
        p = self.clf.predict_proba(self._vec(texts))
        full = np.zeros((len(texts), len(LABELS)))
        for j, c in enumerate(self.clf.classes_):
            full[:, LABELS.index(c)] = p[:, j]
        return [LABELS[i] for i in full.argmax(1)], full


IntentEnum = Enum("IntentEnum", {v: v for v in LABELS}, type=str)
IntentOutput: type[BaseModel] = create_model("IntentOutput", intent=(IntentEnum, ...))


class LLMIntent:
    name = "llm_zero_shot"

    def __init__(self, llm):
        self.llm = llm
        self.errors = 0

    def _prompt(self, text: str) -> list[dict]:
        cats = "\n".join(f"- {i.value}: {d}" for i, d in DESCRIPTIONS.items())
        return [
            {"role": "system", "content": "Ты классифицируешь обращения в контакт-центр страховой компании."},
            {
                "role": "user",
                "content": f"Категории:\n{cats}\n\nРасшифровка звонка:\n{text}\n\n"
                "Определи главную цель клиента. Ответь JSON с полем intent.",
            },
        ]

    def predict_proba(self, texts):
        from cci.llm.client import LLMError

        labels, rows = [], np.zeros((len(texts), len(LABELS)))
        for i, t in enumerate(texts):
            try:
                out = self.llm.structured(self._prompt(t), IntentOutput, max_tokens=32)
                lab = out.intent.value
            except LLMError:
                self.errors += 1
                lab = Intent.GENERAL_INFO.value  # counted as an error; the case goes to review
            labels.append(lab)
            rows[i, LABELS.index(lab)] = 1.0
        return labels, rows
