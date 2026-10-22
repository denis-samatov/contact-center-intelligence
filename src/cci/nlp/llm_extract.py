"""LLM case extraction with a JSON-schema constrained output (the neural counterpart of
:mod:`cci.nlp.rules`)."""

from __future__ import annotations

import datetime as dt

from pydantic import BaseModel, Field

from cci.domain.schema import FIELDS, Contradiction, Preferences
from cci.llm.client import LLMClient, LLMError
from cci.nlp.rules import Extraction


class _Fields(BaseModel):
    policy_number: str | None = Field(None, description="8 цифр без пробелов")
    claim_number: str | None = Field(None, description="6 цифр без пробелов")
    incident_date: str | None = Field(None, description="дата ГГГГ-ММ-ДД")
    incident_place: str | None = None
    address: str | None = None
    damage_type: str | None = Field(None, description="flood | fire | theft")
    injuries: str | None = Field(None, description="yes | no")
    other_party: str | None = Field(None, description="yes | no")
    amount: str | None = Field(None, description="сумма в рублях, только цифры")
    product: str | None = Field(None, description="osago | kasko | property | travel")
    change_type: str | None = Field(None, description="address | phone | drivers")
    new_value: str | None = None
    document_type: str | None = Field(None, description="policy_copy | payment_certificate | claim_decision")
    cancellation_reason: str | None = None
    complaint_subject: str | None = Field(None, description="delay | amount | service")
    question_topic: str | None = Field(None, description="deductible | coverage | documents | deadlines")


class _Contra(BaseModel):
    field: str
    values: list[str]


class LLMCase(BaseModel):
    fields: _Fields
    contradictions: list[_Contra] = Field(
        default_factory=list, description="поля, где клиент назвал разные значения БЕЗ явного исправления"
    )
    preferred_channel: str | None = Field(None, description="sms | email | phone, только если клиент сам сказал")
    callback_window: str | None = Field(None, description="morning | afternoon | evening, только если сказал")


def _prompt(transcript: str, call_date: dt.date | None) -> list[dict]:
    field_doc = "\n".join(f"- {k}: {v}" for k, v in FIELDS.items())
    return [
        {
            "role": "system",
            "content": "Ты извлекаешь данные из расшифровок звонков в страховую компанию. "
            "Не придумывай значения: если клиент что-то не назвал, ставь null.",
        },
        {
            "role": "user",
            "content": (
                f"Дата звонка: {call_date.isoformat() if call_date else 'неизвестна'}.\n"
                f"Поля:\n{field_doc}\n\n"
                "Числа, сказанные словами, переводи в цифры. Если клиент исправился («не X, а Y»), бери исправленное. "
                "Если клиент назвал разные значения без исправления, добавь поле в contradictions.\n\n"
                f"Расшифровка:\n{transcript}\n\nОтветь JSON."
            ),
        },
    ]


def llm_extract(llm: LLMClient, turns: list[tuple[str, str]], call_date: dt.date | None) -> Extraction:
    transcript = "\n".join(f"{r}: {t}" for r, t in turns)
    try:
        case = llm.structured(_prompt(transcript, call_date), LLMCase, max_tokens=400)
    except LLMError:
        return Extraction(unanswered=["__llm_error__"])
    fields = {k: str(v).strip() for k, v in case.fields.model_dump().items() if v not in (None, "", "null")}
    prefs = Preferences(
        channel=case.preferred_channel if case.preferred_channel in ("sms", "email", "phone") else None,
        callback_window=case.callback_window if case.callback_window in ("morning", "afternoon", "evening") else None,
    )
    contra = [Contradiction(field=c.field, values=c.values) for c in case.contradictions if c.field in FIELDS]
    return Extraction(fields=fields, contradictions=contra, preferences=prefs)
