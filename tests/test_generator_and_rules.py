import collections
import datetime as dt

from cci.domain.schema import REQUIRED, Intent
from cci.nlp.rules import extract
from cci.synth.generator import SPLIT_FAMILIES, generate_corpus


def test_corpus_is_deterministic_and_family_disjoint():
    a = generate_corpus(50, 20, 20, seed=1)
    b = generate_corpus(50, 20, 20, seed=1)
    assert [c.to_json() for c in a] == [c.to_json() for c in b]
    fam = collections.defaultdict(set)
    for c in a:
        fam[c.split].add(c.family)
    assert fam["train"] <= set(SPLIT_FAMILIES["train"])
    assert fam["test"] == {"D"} and not fam["test"] & fam["train"]


def test_gold_labels_are_consistent():
    for c in generate_corpus(0, 0, 200, seed=3):
        lab = c.label
        it = Intent(lab["intent"])
        assert set(lab["missing_required"]) == {f for f in REQUIRED[it] if f not in lab["fields"]}
        assert set(lab["missing_required"]) == set(c.phenomena["missing"])
        for contra in lab["contradictions"]:
            assert len(set(contra["values"])) == 2


def test_rules_handle_corrections_and_contradictions():
    turns = [
        ("operator", "Назовите, пожалуйста, номер полиса."),
        ("client", "Номер полиса один два три четыре пять шесть семь восемь."),
        (
            "client",
            "Ой, извините, ошибся, не один два три четыре пять шесть семь восемь, а восемь семь шесть пять четыре три два один.",
        ),
        ("operator", "Когда это произошло?"),
        ("client", "Это было пятнадцатого октября."),
        ("operator", "Что-нибудь еще хотите добавить?"),
        ("client", "Да, и еще раз, это было шестнадцатого октября, если что."),
    ]
    ex = extract(turns, dt.date(2026, 10, 20))
    assert ex.fields["policy_number"] == "87654321"  # correction applied, not a contradiction
    assert ex.fields["incident_date"] == "2026-10-15"
    assert [c.field for c in ex.contradictions] == ["incident_date"]


def test_rules_report_unanswered_fields():
    turns = [("operator", "Назовите номер убытка, пожалуйста."), ("client", "Ой, не помню, он дома остался.")]
    assert extract(turns).unanswered == ["claim_number"]


def test_dates_after_call_date_roll_back_a_year():
    turns = [("operator", "Какого числа это случилось?"), ("client", "Тридцать первого декабря.")]
    assert extract(turns, dt.date(2026, 10, 5)).fields["incident_date"] == "2025-12-31"
