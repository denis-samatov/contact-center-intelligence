"""Seeded generator of synthetic insurance calls with gold annotations.

Every call is produced from a structured *case* (intent, field values, preferences), then
rendered as an operator/client dialogue in spoken Russian (numbers as words, so TTS reads
them naturally and the reference transcript matches what an ASR system should output).

Phenomena injected with fixed probabilities, each recorded in the gold label:

* **missing** required field: the client cannot provide it ("не помню");
* **correction**: the client states a value, then explicitly corrects it ("ой, не X, а Y");
  gold keeps the corrected value and no contradiction;
* **contradiction**: the client states two different values without a correction cue;
  gold records the contradiction and the field is excluded from value scoring;
* **explicit preferences**: preferred channel / callback window, only when said.

All names, numbers and the insurer are fictional. Nothing is derived from real calls.
"""

from __future__ import annotations

import datetime as dt
import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path

from cci.domain.ru_numbers import spell_date, spell_digits, spell_int
from cci.domain.schema import (
    REQUIRED,
    CaseLabel,
    Contradiction,
    Intent,
    Preferences,
    fields_for,
    missing_required,
    route,
)
from cci.synth import templates as T

SPLIT_FAMILIES = {"train": ("A", "B"), "dev": ("C",), "test": ("D",)}
P_MISSING, P_CORRECTION, P_CONTRADICTION, P_PREFERENCE = 0.15, 0.12, 0.12, 0.45

CHANNEL_PHRASES = {
    "sms": "Лучше пришлите эсэмэс.",
    "email": "Удобнее всего на электронную почту.",
    "phone": "Лучше перезвоните мне.",
}
WINDOW_PHRASES = {
    "morning": "Звоните, пожалуйста, утром.",
    "afternoon": "Мне удобно днем.",
    "evening": "Только вечером, днем я на работе.",
}
PRODUCT_WORDS = {
    "osago": "осаго",
    "kasko": "каско",
    "property": "страхование квартиры",
    "travel": "страховка для путешествий",
}
DAMAGE_WORDS = {"flood": "залив", "fire": "пожар", "theft": "кража"}
CHANGE_WORDS = {"address": "адрес", "phone": "номер телефона", "drivers": "список водителей"}
DOC_WORDS = {
    "policy_copy": "копия полиса",
    "payment_certificate": "справка о выплате",
    "claim_decision": "решение по убытку",
}
SUBJECT_WORDS = {
    "delay": "сроки, всё очень долго",
    "amount": "сумма выплаты слишком маленькая",
    "service": "отношение сотрудников",
}
TOPIC_WORDS = {
    "deductible": "франшиза",
    "coverage": "что покрывается",
    "documents": "какие нужны документы",
    "deadlines": "сроки обращения",
}
REASONS = ["продал машину", "нашел вариант дешевле", "переезжаю", "больше не нужна"]


@dataclass
class Turn:
    speaker: str  # "operator" | "client"
    text: str


@dataclass
class Call:
    call_id: str
    split: str
    family: str
    call_date: str
    operator: str
    turns: list[Turn]
    label: dict
    phenomena: dict = field(default_factory=dict)

    def transcript(self) -> str:
        return "\n".join(f"{t.speaker}: {t.text}" for t in self.turns)

    def to_json(self) -> dict:
        d = asdict(self)
        d["turns"] = [asdict(t) for t in self.turns]
        return d


def _value(fld: str, intent: Intent, rng: random.Random, call_date: dt.date) -> tuple[str, str]:
    """Return (gold value, spoken form)."""
    if fld == "policy_number":
        v = "".join(rng.choice("0123456789") for _ in range(8))
        return v, spell_digits(v)
    if fld == "claim_number":
        v = f"{rng.randint(100, 999)}{rng.randint(100, 999)}"
        spoken = f"{spell_int(int(v[:3]))} {spell_int(int(v[3:]))}" if rng.random() < 0.6 else spell_digits(v)
        return v, spoken
    if fld == "incident_date":
        d = call_date - dt.timedelta(days=rng.randint(1, 25))
        return d.isoformat(), spell_date(d.day, d.month)
    if fld == "incident_place":
        v = rng.choice(T.CITIES + T.ROADS)
        return v, v
    if fld == "address":
        street, house = rng.choice(T.STREETS), rng.randint(1, 120)
        return f"{street} {house}", f"{street} дом {spell_int(house)}"
    if fld == "damage_type":
        v = rng.choice(list(DAMAGE_WORDS))
        return v, DAMAGE_WORDS[v]
    if fld in ("injuries", "other_party"):
        v = rng.choice(["yes", "no"]) if fld == "other_party" else ("yes" if rng.random() < 0.25 else "no")
        words = {
            ("injuries", "yes"): "Да, есть пострадавший, водитель ударился.",
            ("injuries", "no"): "Нет, все целы, никто не пострадал.",
            ("other_party", "yes"): "Да, был второй автомобиль.",
            ("other_party", "no"): "Нет, я один был, въехал в ограждение.",
        }
        return v, words[(fld, v)]
    if fld == "amount":
        v = rng.choice([15, 30, 45, 80, 120, 250, 400]) * 1000
        return str(v), f"примерно {spell_int(v)} рублей"
    if fld == "product":
        v = {"policy_renewal": rng.choice(["osago", "kasko", "property"])}.get(
            intent.value, rng.choice(list(PRODUCT_WORDS))
        )
        return v, PRODUCT_WORDS[v]
    if fld == "change_type":
        v = rng.choice(list(CHANGE_WORDS))
        return v, CHANGE_WORDS[v]
    if fld == "document_type":
        v = rng.choice(list(DOC_WORDS))
        return v, DOC_WORDS[v]
    if fld == "cancellation_reason":
        v = rng.choice(REASONS)
        return v, v
    if fld == "complaint_subject":
        v = rng.choice(list(SUBJECT_WORDS))
        return v, SUBJECT_WORDS[v]
    if fld == "question_topic":
        v = rng.choice(list(TOPIC_WORDS))
        return v, TOPIC_WORDS[v]
    raise KeyError(fld)


def _new_value_for_change(change_type: str, rng: random.Random) -> tuple[str, str]:
    if change_type == "phone":
        v = "9" + "".join(rng.choice("0123456789") for _ in range(9))
        return v, spell_digits(v)
    if change_type == "address":
        street, house = rng.choice(T.STREETS), rng.randint(1, 120)
        return f"{street} {house}", f"{street} дом {spell_int(house)}"
    name = rng.choice(["Иванов Петр", "Смирнова Анна", "Козлов Илья", "Попова Мария"])
    return name, f"добавить водителя {name}"


def _answer(fld: str, spoken: str, rng: random.Random) -> str:
    lead = {
        "policy_number": ["Номер полиса", "Да, сейчас, номер", "Записывайте,"],
        "claim_number": ["Номер убытка", "Сейчас, номер", "Да, номер заявления"],
        "incident_date": ["Это было", "Случилось", ""],
        "incident_place": ["Это было в городе", "Это случилось,", "Место -"],
        "address": ["Адрес", "Квартира находится по адресу", ""],
        "damage_type": ["У нас", "Произошел", ""],
        "amount": ["Ущерб", "Думаю,", ""],
        "product": ["У меня", "Это", ""],
        "change_type": ["Нужно поменять", "Хочу изменить", ""],
        "new_value": ["Новые данные:", "Запишите:", ""],
        "document_type": ["Мне нужна", "Нужна", ""],
        "cancellation_reason": ["Причина в том, что я", "Ну я", ""],
        "complaint_subject": ["Меня не устраивает", "Не устраивает", ""],
        "question_topic": ["Интересует", "Хочу узнать про", ""],
    }.get(fld, [""])
    if fld in ("injuries", "other_party"):
        return spoken
    prefix = rng.choice(lead)
    filler = rng.choice(T.FILLERS) + ", " if rng.random() < 0.25 else ""
    return f"{filler}{prefix} {spoken}.".replace("  ", " ").strip().capitalize()


def _other_spoken(fld: str, gold: str, intent: Intent, rng: random.Random, call_date: dt.date) -> tuple[str, str]:
    for _ in range(20):
        v, s = _value(fld, intent, rng, call_date)
        if v != gold:
            return v, s
    raise RuntimeError("could not draw a different value")


def generate_call(idx: int, split: str, rng: random.Random) -> Call:
    intent = rng.choice(list(Intent))
    family = rng.choice(SPLIT_FAMILIES[split])
    call_date = dt.date(2026, 10, 1) + dt.timedelta(days=rng.randint(0, 20))
    op = rng.choice(T.OPERATOR_NAMES)
    turns = [
        Turn("operator", rng.choice(T.GREETINGS).format(op=op)),
        Turn("client", rng.choice(T.OPENINGS[intent.value][family])),
    ]
    gold: dict[str, str] = {}
    contradictions: list[Contradiction] = []
    phen = {"missing": [], "correction": [], "contradiction": []}
    wanted = list(fields_for(intent))
    if intent is Intent.DATA_CHANGE:
        wanted = ["policy_number", "change_type", "new_value"]
    can_miss = [
        f
        for f in REQUIRED[intent]
        if f
        not in (
            "injuries",
            "other_party",
            "product",
            "question_topic",
            "change_type",
            "document_type",
            "damage_type",
            "complaint_subject",
        )
    ]
    miss = rng.choice(can_miss) if can_miss and rng.random() < P_MISSING else None
    for fld in wanted:
        turns.append(Turn("operator", rng.choice(T.ASK[fld])))
        if fld == miss:
            turns.append(Turn("client", rng.choice(T.DONT_KNOW)))
            phen["missing"].append(fld)
            continue
        if fld == "new_value":
            v, spoken = _new_value_for_change(gold.get("change_type", "phone"), rng)
        else:
            v, spoken = _value(fld, intent, rng, call_date)
        r = rng.random()
        if r < P_CORRECTION and fld in ("policy_number", "incident_date", "amount", "claim_number"):
            wrong, wrong_sp = _other_spoken(fld, v, intent, rng, call_date)
            turns.append(Turn("client", _answer(fld, wrong_sp, rng)))
            turns.append(Turn("client", f"Ой, извините, ошибся, не {wrong_sp}, а {spoken}."))
            phen["correction"].append(fld)
        else:
            turns.append(Turn("client", _answer(fld, spoken, rng)))
        gold[fld] = v
        if fld == "amount" and rng.random() < 0.3:
            turns.append(Turn("operator", rng.choice(T.CONFIRM)))
    # Contradiction: a later restatement that disagrees, without a correction cue.
    candidates = [
        f for f in ("incident_date", "injuries", "policy_number") if f in gold and f not in phen["correction"]
    ]
    if gold.get("injuries") == "yes":
        candidates.remove("injuries")  # only the "no -> later hospital visit" direction is a contradiction
    if candidates and rng.random() < P_CONTRADICTION:
        fld = rng.choice(candidates)
        if fld == "injuries":
            other, restate = "yes", "Кстати, мне потом в травмпункт пришлось ехать."
        else:
            other, other_sp = _other_spoken(fld, gold[fld], intent, rng, call_date)
            restate = (
                f"Да, и еще раз, это было {other_sp}, если что."
                if fld == "incident_date"
                else f"Повторю номер полиса: {other_sp}."
            )
        turns.append(Turn("operator", "Что-нибудь еще хотите добавить?"))
        turns.append(Turn("client", restate))
        contradictions.append(Contradiction(field=fld, values=sorted([gold[fld], other])))
        phen["contradiction"].append(fld)
    prefs = Preferences()
    if rng.random() < P_PREFERENCE:
        turns.append(Turn("operator", "Как вам удобнее получать информацию?"))
        if rng.random() < 0.6:
            prefs.channel = rng.choice(list(CHANNEL_PHRASES))
            turns.append(Turn("client", CHANNEL_PHRASES[prefs.channel]))
        else:
            prefs.callback_window = rng.choice(list(WINDOW_PHRASES))
            turns.append(Turn("client", WINDOW_PHRASES[prefs.callback_window]))
    turns.append(Turn("operator", rng.choice(T.CLOSINGS_OP)))
    turns.append(Turn("client", rng.choice(T.CLOSINGS_CLIENT)))
    queue = route(intent, gold)
    label = CaseLabel(
        intent=intent,
        fields=gold,
        preferences=prefs,
        contradictions=contradictions,
        missing_required=missing_required(intent, gold),
        queue=queue,
        next_actions=next_actions(intent, gold, prefs),
    )
    return Call(
        f"{split}-{idx:04d}", split, family, call_date.isoformat(), op, turns, label.model_dump(mode="json"), phen
    )


def next_actions(intent: Intent, fields: dict[str, str], prefs: Preferences) -> list[str]:
    """Business rules for the operator's next steps (used for gold and for the product)."""
    acts = []
    if missing_required(intent, fields):
        acts.append("request_missing_info")
    acts += {
        Intent.CLAIM_AUTO: ["send_document_checklist", "schedule_vehicle_inspection"],
        Intent.CLAIM_PROPERTY: ["send_document_checklist", "schedule_property_assessment"],
        Intent.PAYMENT_STATUS: ["send_payment_status"],
        Intent.POLICY_RENEWAL: ["prepare_renewal_quote"],
        Intent.POLICY_CANCELLATION: ["calculate_refund"],
        Intent.DATA_CHANGE: ["update_policy_record"],
        Intent.COMPLAINT: ["register_complaint"],
        Intent.DOCUMENT_REQUEST: ["issue_document"],
        Intent.GENERAL_INFO: ["send_product_info"],
    }[intent]
    if prefs.channel:
        acts.append(f"notify_via_{prefs.channel}")
    if prefs.callback_window:
        acts.append(f"callback_{prefs.callback_window}")
    return acts


def generate_corpus(n_train: int, n_dev: int, n_test: int, seed: int = 20261022) -> list[Call]:
    calls = []
    for split, n in (("train", n_train), ("dev", n_dev), ("test", n_test)):
        rng = random.Random(f"{seed}-{split}")
        calls += [generate_call(i, split, rng) for i in range(n)]
    return calls


def write_corpus(calls: list[Call], out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calls.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for c in calls:
            fh.write(json.dumps(c.to_json(), ensure_ascii=False) + "\n")
    return path


def load_corpus(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
