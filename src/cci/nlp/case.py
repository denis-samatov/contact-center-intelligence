"""Case assembly: intent + extraction -> case card, protocol, review decision, draft reply.

Selective prediction (automation vs human review) uses observable signals only:

* intent probability below a threshold chosen on the *dev* split for a target error rate;
* missing required fields;
* contradictions;
* LLM/schema failures or ungrounded drafts.

The protocol is rendered deterministically from extracted fields (no generation, so it
cannot hallucinate); the customer reply draft is template-based and cites the KB rules it
relies on. Neither is sent anywhere automatically.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from cci.domain.schema import REQUIRED, Intent, Preferences, fields_for, missing_required, route
from cci.nlp.rules import Extraction
from cci.synth.generator import next_actions

INTENT_RU = {
    "claim_auto": "Страховой случай: автомобиль",
    "claim_property": "Страховой случай: имущество",
    "payment_status": "Статус выплаты",
    "policy_renewal": "Продление полиса",
    "policy_cancellation": "Расторжение договора",
    "data_change": "Изменение данных",
    "complaint": "Жалоба",
    "document_request": "Запрос документа",
    "general_info": "Информационный вопрос",
}
FIELD_RU = {
    "policy_number": "Номер полиса",
    "claim_number": "Номер убытка",
    "incident_date": "Дата происшествия",
    "incident_place": "Место",
    "address": "Адрес",
    "damage_type": "Тип ущерба",
    "injuries": "Пострадавшие",
    "other_party": "Второй участник",
    "amount": "Оценка ущерба, руб.",
    "product": "Продукт",
    "change_type": "Что изменить",
    "new_value": "Новое значение",
    "document_type": "Документ",
    "cancellation_reason": "Причина",
    "complaint_subject": "Предмет жалобы",
    "question_topic": "Тема вопроса",
}
KB_FOR_INTENT = {
    "claim_auto": ["kasko_claims#2", "osago_claims#2"],
    "claim_property": ["property_claims#2", "property_claims#3"],
    "payment_status": ["payments#1", "payments#2"],
    "policy_renewal": ["service#4"],
    "policy_cancellation": ["cancellation#2", "cancellation#3"],
    "data_change": ["service#1", "service#2"],
    "complaint": ["complaints#1", "complaints#2"],
    "document_request": ["service#3", "payments#3"],
    "general_info": [],
}


@dataclass
class ReviewPolicy:
    min_intent_prob: float = 0.6
    review_on_missing: bool = True
    review_on_contradiction: bool = True


@dataclass
class Case:
    intent: str
    intent_prob: float
    fields: dict[str, str]
    missing: list[str]
    contradictions: list[dict]
    preferences: dict
    queue: str
    next_actions: list[str]
    decision: str  # "auto" | "review"
    review_reasons: list[str] = field(default_factory=list)
    protocol: str = ""
    reply_draft: str = ""
    reply_sources: list[str] = field(default_factory=list)


def build_case(intent: str, prob: float, ex: Extraction, policy: ReviewPolicy) -> Case:
    it = Intent(intent)
    allowed = set(fields_for(it))
    fields = {k: v for k, v in ex.fields.items() if k in allowed}
    missing = missing_required(it, fields)
    reasons = []
    if prob < policy.min_intent_prob:
        reasons.append("low_intent_confidence")
    if missing and policy.review_on_missing:
        reasons.append("missing_required_fields")
    contra = [c.model_dump() for c in ex.contradictions if c.field in allowed]
    if contra and policy.review_on_contradiction:
        reasons.append("contradiction")
    if "__llm_error__" in ex.unanswered:
        reasons.append("extraction_failed")
    prefs = ex.preferences if isinstance(ex.preferences, Preferences) else Preferences(**ex.preferences)
    case = Case(
        intent=intent,
        intent_prob=round(float(prob), 4),
        fields=fields,
        missing=missing,
        contradictions=contra,
        preferences=prefs.model_dump(),
        queue=route(it, fields).value,
        next_actions=next_actions(it, fields, prefs),
        decision="review" if reasons else "auto",
        review_reasons=reasons,
    )
    case.protocol = render_protocol(case)
    case.reply_draft, case.reply_sources = draft_reply(case)
    return case


def render_protocol(case: Case) -> str:
    lines = [f"Тема обращения: {INTENT_RU[case.intent]} (уверенность {case.intent_prob:.2f})"]
    for f in REQUIRED[Intent(case.intent)] + tuple(k for k in case.fields if k not in REQUIRED[Intent(case.intent)]):
        lines.append(f"- {FIELD_RU[f]}: {case.fields.get(f, 'НЕ УКАЗАНО')}")
    if case.contradictions:
        lines.append(
            "Противоречия: "
            + "; ".join(f"{FIELD_RU[c['field']]}: {' / '.join(c['values'])}" for c in case.contradictions)
        )
    prefs = [v for v in case.preferences.values() if v]
    if prefs:
        lines.append("Пожелания клиента: " + ", ".join(prefs))
    lines.append(f"Очередь: {case.queue}. Следующие шаги: {', '.join(case.next_actions)}.")
    decision_text = {
        "auto": "автоматически",
        "review": "на проверку оператору",
        "approved": "одобрено оператором",
        "rejected": "отклонено оператором",
    }[case.decision]
    lines.append(
        f"Решение системы: {decision_text}" + (f" ({', '.join(case.review_reasons)})" if case.review_reasons else "")
    )
    return "\n".join(lines)


def draft_reply(case: Case) -> tuple[str, list[str]]:
    sources = KB_FOR_INTENT.get(case.intent, [])
    parts = ["Здравствуйте! Ваше обращение зарегистрировано."]
    if case.missing:
        parts.append(
            "Чтобы продолжить, пожалуйста, сообщите: " + ", ".join(FIELD_RU[m].lower() for m in case.missing) + "."
        )
    if sources:
        parts.append("Порядок действий и сроки описаны в правилах: " + ", ".join(sources) + ".")
    parts.append("С уважением, Демо-Страхование.")
    return " ".join(parts), sources
