"""Rule-based case extraction: the classical baseline for the LLM extractor.

Input is a list of role-labelled turns (from reference transcripts, or from ASR +
diarization + role assignment). The extractor exploits the dialogue structure of a contact
center: an operator question announces which field the next client turn answers.

* Values are parsed with :mod:`cci.domain.ru_numbers` (digits, dates, amounts) and closed
  gazetteers (cities, streets, products).
* An explicit correction ("не X, а Y") replaces the value; it is *not* a contradiction.
* Two different values for the same field without a correction cue are reported as a
  contradiction; the first value is kept and the case is flagged for review.

Every extracted value keeps the turn index it came from (evidence for the UI).
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass, field

from cci.domain.ru_numbers import parse_numbers
from cci.domain.schema import Contradiction, Preferences
from cci.synth import templates as T

QUESTION_CUES: list[tuple[str, re.Pattern]] = [
    ("policy_number", re.compile(r"номер (вашего )?(полиса|договора)")),
    ("claim_number", re.compile(r"номер (убытка|вашего заявления|заявления)")),
    ("incident_date", re.compile(r"когда это|какого числа|дату происшеств")),
    ("incident_place", re.compile(r"где это|место происшеств|в каком городе")),
    ("address", re.compile(r"адрес")),
    ("damage_type", re.compile(r"что именно произошло|характер повреждения")),
    ("injuries", re.compile(r"пострадавш|кто-нибудь пострадал")),
    ("other_party", re.compile(r"второй участник|другой автомобиль")),
    ("amount", re.compile(r"оценить ущерб|какую сумму")),
    ("product", re.compile(r"вид страхования|какой это продукт")),
    ("change_type", re.compile(r"что именно нужно изменить")),
    ("new_value", re.compile(r"новое значение|новые данные")),
    ("document_type", re.compile(r"какой именно документ")),
    ("cancellation_reason", re.compile(r"причину расторжения")),
    ("complaint_subject", re.compile(r"что именно вас не устраивает")),
    ("question_topic", re.compile(r"что именно вас интересует")),
]
CORRECTION = re.compile(r"(ошиб|извините|перепутал).*?\bне\b(?P<wrong>.+?),?\s+а\s+(?P<right>.+)$")
DONT_KNOW = re.compile(r"не помню|не могу сказать|не знаю|нет под рукой")

KEYWORDS = {
    "damage_type": [
        ("flood", r"зали|затоп|теч|трубу|залив"),
        ("fire", r"пожар|возгор|загор|сажа"),
        ("theft", r"краж|обокр|вынесли|вскрыли|пропали"),
    ],
    "product": [("osago", r"осаго"), ("kasko", r"каско"), ("property", r"квартир|имуществ"), ("travel", r"путешеств")],
    "change_type": [("phone", r"телефон"), ("address", r"адрес"), ("drivers", r"водител")],
    "document_type": [
        ("policy_copy", r"копи|дубликат|полис"),
        ("payment_certificate", r"справк|выплат"),
        ("claim_decision", r"решени"),
    ],
    "complaint_subject": [
        ("delay", r"срок|долго|тянете|ждать"),
        ("amount", r"сумм|маленьк|меньше"),
        ("service", r"отношени|сотрудник|бросил"),
    ],
    "question_topic": [
        ("deductible", r"франшиз"),
        ("documents", r"документ"),
        ("deadlines", r"срок"),
        ("coverage", r"покрыва|входит"),
    ],
}


@dataclass
class Extraction:
    fields: dict[str, str] = field(default_factory=dict)
    evidence: dict[str, int] = field(default_factory=dict)  # field -> turn index
    contradictions: list[Contradiction] = field(default_factory=list)
    preferences: Preferences = field(default_factory=Preferences)
    unanswered: list[str] = field(default_factory=list)


def _norm(t: str) -> str:
    return t.lower().replace("ё", "е")


def _keyword(fld: str, text: str) -> str | None:
    for value, pat in KEYWORDS[fld]:
        if re.search(pat, text):
            return value
    return None


def _resolve_date(mm_dd: str, call_date: dt.date | None) -> str:
    month, day = (int(x) for x in mm_dd.split("-"))
    year = (call_date or dt.date.today()).year
    d = dt.date(year, month, day)
    if call_date and d > call_date:  # incidents are in the past
        d = dt.date(year - 1, month, day)
    return d.isoformat()


def parse_value(fld: str, text: str, call_date: dt.date | None = None) -> str | None:
    t = _norm(text)
    nums = parse_numbers(t)
    if fld == "policy_number":
        cands = [n.text for n in nums if n.kind in ("digits", "grouped", "cardinal") and len(n.text) == 8]
        return cands[-1] if cands else None
    if fld == "claim_number":
        cands = [n.text for n in nums if len(n.text) == 6]
        return cands[-1] if cands else None
    if fld == "new_value":
        phone = [n.text for n in nums if len(n.text) == 10]
        if phone:
            return phone[-1]
        for street in T.STREETS:
            if _norm(street) in t:
                house = [n.text for n in nums if n.kind == "cardinal" and len(n.text) <= 3]
                return f"{street} {house[-1]}" if house else street
        m = re.search(r"водителя\s+(.+?)[.,]?$", text.strip())
        return m.group(1).strip(" .") if m else None
    if fld == "incident_date":
        dates = [n.text for n in nums if n.kind == "date"]
        return _resolve_date(dates[-1], call_date) if dates else None
    if fld == "incident_place":
        for place in T.CITIES + T.ROADS:
            if _norm(place) in t or _norm(place)[:-1] in t:  # "Москве", "Казани" (crude stem)
                return place
        return None
    if fld == "address":
        for street in T.STREETS:
            if _norm(street) in t:
                house = [n.text for n in nums if n.kind == "cardinal" and len(n.text) <= 3]
                return f"{street} {house[-1]}" if house else None
        return None
    if fld == "injuries":
        if re.search(r"никто не пострадал|все целы|не пострадал", t):
            return "no"
        if re.search(r"пострадавш|ударил|травмпункт|больниц|скорая", t):
            return "yes"
        return None
    if fld == "other_party":
        if re.search(r"один был|ограждени|сам ", t):
            return "no"
        if re.search(r"второй автомобил|другой машин|въехал в меня|был второй", t):
            return "yes"
        return None
    if fld == "amount":
        cands = [int(n.text) for n in nums if n.kind == "cardinal" and n.text.isdigit() and int(n.text) >= 1000]
        return str(cands[-1]) if cands else None
    if fld == "cancellation_reason":
        stripped = re.sub(r"^((ну|так|э|вот|значит|сейчас),?\s*)*(причина в том, что я|ну я)?\s*", "", t).strip(" .")
        return stripped or None
    if fld in KEYWORDS:
        return _keyword(fld, t)
    return None


def _question_field(text: str) -> str | None:
    t = _norm(text)
    for fld, pat in QUESTION_CUES:
        if pat.search(t):
            return fld
    return None


def extract(turns: list[tuple[str, str]], call_date: dt.date | None = None) -> Extraction:
    """``turns`` = [(role, text)], role in {"operator", "client"}."""
    ex = Extraction()
    values: dict[str, list[tuple[str, int, bool]]] = {}  # field -> [(value, turn, corrected)]
    pending: str | None = None
    last_answered: str | None = None  # a correction refers to the most recent answer
    for i, (role, text) in enumerate(turns):
        t = _norm(text)
        if role == "operator":
            q = _question_field(text)
            if q:
                pending = q
            elif "что-нибудь еще" in t:
                pending = "__restatement__"
            continue
        if role != "client":
            continue
        corr = CORRECTION.search(t)
        if corr and last_answered and values.get(last_answered):
            v = parse_value(last_answered, corr.group("right"), call_date)
            if v:
                values[last_answered] = [(v, i, True)]
            continue
        if pending == "__restatement__":
            # A restatement may concern any field; parse the ones the turn mentions.
            for fld in ("incident_date", "policy_number", "injuries"):
                v = parse_value(fld, text, call_date)
                if v and fld in values:
                    values[fld].append((v, i, False))
            continue
        if pending:
            if DONT_KNOW.search(t):
                ex.unanswered.append(pending)
            else:
                v = parse_value(pending, text, call_date)
                if v:
                    values.setdefault(pending, []).append((v, i, False))
                    last_answered = pending
            pending = None
        # Preferences are explicit statements anywhere in client speech.
        if re.search(r"эсэмэс|смс|sms", t):
            ex.preferences.channel = "sms"
        elif re.search(r"почт", t):
            ex.preferences.channel = "email"
        elif re.search(r"перезвоните", t):
            ex.preferences.channel = "phone"
        for window, pat in (("morning", r"утром"), ("afternoon", r"днем"), ("evening", r"вечером")):
            if re.search(pat, t) and re.search(r"звоните|удобно|только", t):
                ex.preferences.callback_window = window
                break
    # Fields that can be inferred from the client's opening when never asked explicitly.
    opening = " ".join(_norm(t) for r, t in turns[:3] if r == "client")
    for fld in ("damage_type", "product", "change_type", "document_type", "complaint_subject", "question_topic"):
        if fld not in values:
            v = _keyword(fld, opening)
            if v:
                values[fld] = [(v, 1, False)]
    for fld, vals in values.items():
        distinct = sorted({v for v, _, _ in vals})
        ex.fields[fld] = vals[0][0]
        ex.evidence[fld] = vals[0][1]
        if len(distinct) > 1:
            ex.contradictions.append(Contradiction(field=fld, values=distinct))
    return ex
