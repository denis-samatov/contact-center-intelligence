import pytest

pytest.importorskip("localdesk")

from cci.nlp.intent import LABELS  # noqa: E402
from cci.nlp.localdesk_intent import LocaldeskIntent  # noqa: E402

WORDS = {
    "claim_auto": ["авария машина", "дтп бампер", "помяли автомобиль", "разбили фару"],
    "claim_property": ["залило квартиру", "пожар в доме", "украли технику", "затопили соседи"],
    "payment_status": ["где моя выплата", "статус выплаты", "когда придут деньги", "выплата задерживается"],
    "policy_renewal": ["продлить полис", "заканчивается страховка", "остаться на год", "новый полис"],
    "policy_cancellation": ["расторгнуть договор", "отказаться от полиса", "вернуть премию", "прекратить страховку"],
    "data_change": ["сменить адрес", "новый телефон", "добавить водителя", "изменить данные"],
    "complaint": ["хочу пожаловаться", "недоволен сроками", "сумма слишком мала", "претензия к сервису"],
    "document_request": ["копия полиса", "справка о выплате", "нужен документ", "решение по убытку"],
    "general_info": ["что покрывает", "какая франшиза", "какие сроки", "какие документы нужны"],
}


def test_localdesk_adapter_trains_with_repeated_templates_and_abstains(tmp_path):
    texts, labels = [], []
    for lab, phrases in WORDS.items():
        for p in phrases[:3]:
            texts += [f"здравствуйте {p}"] * 4  # repeated template openings, as in the corpus
            labels += [lab] * 4
    dev = [f"добрый день {WORDS[lab][3]}" for lab in LABELS]
    model = LocaldeskIntent(workdir=tmp_path).fit(texts, labels, dev, list(LABELS))
    pred, probs = model.predict_proba(["здравствуйте залило квартиру"])
    assert pred[0] == "claim_property" and abs(probs.sum() - 1.0) < 1e-6
    decisions = model.decide(["здравствуйте залило квартиру", "просто вопрос"])
    assert {d["action"] for d in decisions} <= {"route", "review"}
    assert model.report["run_id"] and (tmp_path / "registry").exists()
