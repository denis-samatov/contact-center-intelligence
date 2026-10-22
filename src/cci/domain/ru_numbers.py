"""Russian numerals: spelling (for synthesis) and parsing (for ASR post-processing).

ASR models for Russian emit numbers as words ("пятнадцатого октября", "ноль один два").
Extraction needs digits, so this module converts both ways:

* :func:`spell_int`, :func:`spell_digits`, :func:`spell_date` - used by the synthetic
  dialogue generator so that TTS reads numbers naturally.
* :func:`parse_numbers` - finds number expressions in a transcript and returns digit
  strings with their character spans; used by rule-based extraction.

Coverage is what the contact-center domain needs (cardinals < 10^9, day ordinals, digit
sequences), not general Russian morphology.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
TEENS = [
    "десять",
    "одиннадцать",
    "двенадцать",
    "тринадцать",
    "четырнадцать",
    "пятнадцать",
    "шестнадцать",
    "семнадцать",
    "восемнадцать",
    "девятнадцать",
]
TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]
MONTHS_GEN = [
    "января",
    "февраля",
    "марта",
    "апреля",
    "мая",
    "июня",
    "июля",
    "августа",
    "сентября",
    "октября",
    "ноября",
    "декабря",
]
# Ordinals in the genitive case ("пятнадцатого"), as in "пятнадцатого октября".
DAY_ORD_GEN = [
    "",
    "первого",
    "второго",
    "третьего",
    "четвертого",
    "пятого",
    "шестого",
    "седьмого",
    "восьмого",
    "девятого",
    "десятого",
    "одиннадцатого",
    "двенадцатого",
    "тринадцатого",
    "четырнадцатого",
    "пятнадцатого",
    "шестнадцатого",
    "семнадцатого",
    "восемнадцатого",
    "девятнадцатого",
    "двадцатого",
    "двадцать первого",
    "двадцать второго",
    "двадцать третьего",
    "двадцать четвертого",
    "двадцать пятого",
    "двадцать шестого",
    "двадцать седьмого",
    "двадцать восьмого",
    "двадцать девятого",
    "тридцатого",
    "тридцать первого",
]


def _below_thousand(n: int, feminine: bool = False) -> list[str]:
    words = []
    h, rest = divmod(n, 100)
    if h:
        words.append(HUNDREDS[h])
    if 10 <= rest < 20:
        words.append(TEENS[rest - 10])
    else:
        t, u = divmod(rest, 10)
        if t:
            words.append(TENS[t])
        if u:
            words.append(("одна" if u == 1 else "две" if u == 2 else UNITS[u]) if feminine else UNITS[u])
    return words


def _plural(n: int, one: str, few: str, many: str) -> str:
    if 10 < n % 100 < 20:
        return many
    return {1: one, 2: few, 3: few, 4: few}.get(n % 10, many)


def spell_int(n: int) -> str:
    if n == 0:
        return "ноль"
    if not 0 < n < 10**9:
        raise ValueError("supported range is 0..999,999,999")
    words = []
    millions, rest = divmod(n, 10**6)
    thousands, units = divmod(rest, 1000)
    if millions:
        words += _below_thousand(millions) + [_plural(millions, "миллион", "миллиона", "миллионов")]
    if thousands:
        words += _below_thousand(thousands, feminine=True) + [_plural(thousands, "тысяча", "тысячи", "тысяч")]
    if units:
        words += _below_thousand(units)
    return " ".join(words)


def spell_digits(s: str) -> str:
    return " ".join(UNITS[int(c)] for c in s if c.isdigit())


def spell_date(day: int, month: int) -> str:
    return f"{DAY_ORD_GEN[day]} {MONTHS_GEN[month - 1]}"


# ------------------------------------------------------------------------- parsing
_UNIT = {w: i for i, w in enumerate(UNITS)} | {"одна": 1, "две": 2, "одну": 1}
_TEEN = {w: 10 + i for i, w in enumerate(TEENS)}
_TEN = {w: 10 * i for i, w in enumerate(TENS) if w}
_HUND = {w: 100 * i for i, w in enumerate(HUNDREDS) if w}
_SCALE = {
    "тысяча": 1000,
    "тысячи": 1000,
    "тысяч": 1000,
    "тыщ": 1000,
    "тысячу": 1000,
    "миллион": 10**6,
    "миллиона": 10**6,
    "миллионов": 10**6,
}
_DAY = {w: i for i, w in enumerate(DAY_ORD_GEN) if w and " " not in w}
_DAY_TENS = {"двадцать": 20, "тридцать": 30}
_MONTH = {m: i + 1 for i, m in enumerate(MONTHS_GEN)}
NUMBER_WORDS = set(_UNIT) | set(_TEEN) | set(_TEN) | set(_HUND) | set(_SCALE)
_TOKEN = re.compile(r"[а-яё]+|\d+", re.IGNORECASE)


@dataclass(frozen=True)
class NumberSpan:
    text: str  # digits
    start: int  # char offsets in the source text
    end: int
    kind: str  # "cardinal" | "digits" | "date"


def _tokens(text: str) -> list[tuple[str, int, int]]:
    return [(m.group().lower().replace("ё", "е"), m.start(), m.end()) for m in _TOKEN.finditer(text)]


def _parse_cardinal(words: list[str]) -> int | None:
    total, group, seen = 0, 0, False
    for w in words:
        if w in _UNIT:
            group += _UNIT[w]
        elif w in _TEEN:
            group += _TEEN[w]
        elif w in _TEN:
            group += _TEN[w]
        elif w in _HUND:
            group += _HUND[w]
        elif w in _SCALE:
            total += (group or 1) * _SCALE[w]
            group = 0
        else:
            return None
        seen = True
    return total + group if seen else None


def _rank(w: str) -> int:
    """Position of a word inside a sub-thousand group: hundreds 3, tens 2, units/teens 1."""
    if w in _HUND:
        return 3
    if w in _TEN:
        return 2
    return 1


def _split_groups(words: list[str]) -> list[list[str]]:
    """Split a run of number words into separately spoken numbers.

    "сто двадцать три четыреста пятьдесят шесть" -> [[сто двадцать три], [четыреста ...]]:
    within one number the order is hundreds -> tens -> units, so a word whose rank is not
    lower than the previous word's (and no scale word in between) starts a new number.
    Teens and units end a group. Scale words ("тысяч") attach to the current number.
    """
    groups: list[list[str]] = [[]]
    last = 4
    for w in words:
        if w in _SCALE:
            groups[-1].append(w)
            last = 4
            continue
        r = _rank(w)
        if groups[-1] and (r >= last or last == 0):
            groups.append([])
        groups[-1].append(w)
        last = 0 if (w in _TEEN or r == 1) else r
    return [g for g in groups if g]


def _is_digit_sequence(words: list[str]) -> bool:
    """'ноль один два' style: every word is a single digit (or a tens word in a group)."""
    return len(words) >= 4 and all(w in _UNIT for w in words)


def parse_numbers(text: str) -> list[NumberSpan]:
    """Find number expressions. Digit sequences (>= 4 single digits) are kept as strings so
    that leading zeros in policy numbers survive."""
    toks = _tokens(text)
    out: list[NumberSpan] = []
    i = 0
    while i < len(toks):
        w = toks[i][0]
        if w.isdigit():
            out.append(NumberSpan(w, toks[i][1], toks[i][2], "cardinal"))
            i += 1
            continue
        # day ordinal + month  -> date (month-day only; year is resolved by the caller)
        day = None
        j = i
        if w in _DAY_TENS and i + 1 < len(toks) and toks[i + 1][0] in _DAY and _DAY[toks[i + 1][0]] < 10:
            day, j = _DAY_TENS[w] + _DAY[toks[i + 1][0]], i + 2
        elif w in _DAY:
            day, j = _DAY[w], i + 1
        if day is not None and j < len(toks) and toks[j][0] in _MONTH:
            out.append(NumberSpan(f"{_MONTH[toks[j][0]]:02d}-{day:02d}", toks[i][1], toks[j][2], "date"))
            i = j + 1
            continue
        if w in NUMBER_WORDS:
            j = i
            while j < len(toks) and toks[j][0] in NUMBER_WORDS:
                j += 1
            words = [t[0] for t in toks[i:j]]
            if _is_digit_sequence(words):
                out.append(NumberSpan("".join(str(_UNIT[x]) for x in words), toks[i][1], toks[j - 1][2], "digits"))
            else:
                groups = _split_groups(words)
                values = [_parse_cardinal(g) for g in groups]
                if all(v is not None for v in values):
                    if len(values) == 1:
                        out.append(NumberSpan(str(values[0]), toks[i][1], toks[j - 1][2], "cardinal"))
                    else:  # dictated in groups: keep the digit string ("123 456" -> "123456")
                        out.append(NumberSpan("".join(str(v) for v in values), toks[i][1], toks[j - 1][2], "grouped"))
            i = j
            continue
        i += 1
    return out


def normalize_numbers(text: str) -> str:
    """Replace number expressions with digits (dates become MM-DD)."""
    spans = parse_numbers(text)
    out, pos = [], 0
    for s in spans:
        out.append(text[pos : s.start])
        out.append(s.text)
        pos = s.end
    out.append(text[pos:])
    return "".join(out)
