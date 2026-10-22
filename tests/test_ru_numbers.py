import pytest

from cci.domain.ru_numbers import normalize_numbers, parse_numbers, spell_date, spell_digits, spell_int


@pytest.mark.parametrize(
    "n",
    [
        0,
        1,
        2,
        4,
        5,
        11,
        19,
        20,
        21,
        99,
        100,
        101,
        999,
        1000,
        1001,
        2000,
        2500,
        21000,
        120000,
        400000,
        1_000_000,
        1_234_567,
        999_999_999,
    ],
)
def test_cardinal_round_trip(n):
    spans = parse_numbers(spell_int(n))
    assert [s.text for s in spans] == [str(n)]


def test_digit_sequences_keep_leading_zeros():
    assert [s.text for s in parse_numbers("полис " + spell_digits("00123456"))] == ["00123456"]


@pytest.mark.parametrize("day,month", [(1, 1), (9, 3), (15, 10), (21, 12), (23, 10), (31, 8)])
def test_dates(day, month):
    s = parse_numbers("это было " + spell_date(day, month))
    assert s[0].kind == "date" and s[0].text == f"{month:02d}-{day:02d}"


def test_grouped_dictation_is_concatenated():
    assert parse_numbers("сто двадцать три четыреста пятьдесят шесть")[0].text == "123456"
    assert parse_numbers("сто двадцать тысяч рублей")[0].text == "120000"


def test_normalize_numbers_keeps_other_words():
    assert normalize_numbers("примерно сорок пять тысяч рублей") == "примерно 45000 рублей"
