"""Source-grounded answers with citations, and the closed-book baseline.

The grounded prompt gives the model the retrieved chunks with their ids and requires a JSON
answer that cites chunk ids. After generation the answer is *checked*, not trusted:

* citation validity - every cited id must be one of the retrieved chunks;
* numeric grounding - every number in the answer (digits or number words) must occur in
  the cited chunks; otherwise the answer is flagged ``ungrounded`` and routed to a human.

If retrieval returns nothing relevant, the system answers "нет информации" instead of
letting the model guess.
"""

from __future__ import annotations

import re

from pydantic import BaseModel, Field

from cci.domain.ru_numbers import normalize_numbers
from cci.llm.client import LLMClient, LLMError
from cci.rag.retrieval import Chunk, HybridIndex

SYSTEM = (
    "Ты помощник оператора контакт-центра страховой компании «Демо-Страхование». Отвечай кратко, "
    "по-русски, только на основе приведенных правил. Если в правилах нет ответа, ответь 'нет информации'."
)


class GroundedAnswer(BaseModel):
    answer: str = Field(description="Короткий ответ на вопрос")
    citations: list[str] = Field(description="Идентификаторы использованных правил, например kasko_claims#1")


class ClosedBookAnswer(BaseModel):
    answer: str


_NUM = re.compile(r"\d+")


def _numbers(text: str) -> set[str]:
    return set(_NUM.findall(normalize_numbers(text.lower().replace("ё", "е")).replace(" 000", "000")))


def verify(answer: GroundedAnswer, retrieved: list[Chunk]) -> dict:
    ids = {c.chunk_id for c in retrieved}
    cited = [c for c in retrieved if c.chunk_id in answer.citations]
    invalid = [c for c in answer.citations if c not in ids]
    context_numbers = set().union(*(_numbers(c.text) for c in cited)) if cited else set()
    answer_numbers = _numbers(answer.answer)
    unsupported = sorted(answer_numbers - context_numbers)
    no_info = "нет информации" in answer.answer.lower()
    grounded = no_info or (bool(cited) and not invalid and not unsupported)
    return {
        "valid_citations": not invalid,
        "invalid_citations": invalid,
        "unsupported_numbers": unsupported,
        "grounded": grounded,
        "no_info": no_info,
    }


def answer_grounded(llm: LLMClient, index: HybridIndex, question: str, k: int = 3, mode: str = "hybrid") -> dict:
    retrieved = index.search(question, k=k, mode=mode)
    context = "\n".join(f"[{c.chunk_id}] {c.title}. {c.text}" for c in retrieved)
    msgs = [
        {"role": "system", "content": SYSTEM},
        {
            "role": "user",
            "content": f"Правила:\n{context}\n\nВопрос клиента: {question}\nОтветь JSON с полями answer и citations.",
        },
    ]
    try:
        ans = llm.structured(msgs, GroundedAnswer, max_tokens=200)
    except LLMError as e:
        return {"answer": None, "error": str(e), "retrieved": [c.chunk_id for c in retrieved], "grounded": False}
    return {
        "answer": ans.answer,
        "citations": ans.citations,
        "retrieved": [c.chunk_id for c in retrieved],
        **verify(ans, retrieved),
    }


def answer_closed_book(llm: LLMClient, question: str) -> dict:
    msgs = [
        {
            "role": "system",
            "content": "Ты помощник оператора контакт-центра страховой компании "
            "«Демо-Страхование». Отвечай кратко по-русски.",
        },
        {"role": "user", "content": f"Вопрос клиента: {question}\nОтветь JSON с полем answer."},
    ]
    try:
        return {"answer": llm.structured(msgs, ClosedBookAnswer, max_tokens=200).answer}
    except LLMError as e:
        return {"answer": None, "error": str(e)}
