from cci.llm.client import Usage
from cci.synth.paraphrase import FEW_SHOT, paraphrase_calls


class FakeLLM:
    """Returns a scripted paraphrase per input turn."""

    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping
        self.usage = Usage()

    def structured(self, messages, model, max_tokens=None, temperature=None):
        src = messages[-1]["content"].rsplit("Реплика: ", 1)[1]
        return model(text=self.mapping[src])


def _call(*client_turns: str) -> dict:
    return {"call_id": "c1", "turns": [{"speaker": "client", "text": t} for t in client_turns]}


def test_accepts_fact_preserving_rewrite_and_rejects_violations():
    good = "Номер полиса ноль три три девять семь три пять восемь."
    changed = "Примерно сорок пять тысяч рублей."
    copied = "Звоните, пожалуйста, утром."
    same = "Спасибо, до свидания."
    llm = FakeLLM(
        {
            good: "Записывайте номер: ноль три три девять семь три пять восемь.",
            changed: "Около пятидесяти тысяч рублей.",
            copied: FEW_SHOT[0][1],
            same: "спасибо, до свидания",
        }
    )
    res = paraphrase_calls([_call(good, changed, copied, same)], llm)
    assert res["accepted"] == 1
    assert res["rejection_reasons"] == {"numbers_changed": 1, "copied_example": 1, "unchanged": 1}
    turns = [t for _, t in res["calls"][0]["turns"]]
    assert turns[0].startswith("Записывайте номер")
    assert turns[1:] == [changed, copied, same]  # rejected turns keep the original text


def test_similarity_gate_rejects_unrelated_text():
    src = "Удобнее всего на электронную почту."
    llm = FakeLLM({src: "Удобнее всего на электронную почту, а еще у меня кот."})
    vecs = {src: [1.0, 0.0], llm.mapping[src]: [0.0, 1.0]}
    res = paraphrase_calls([_call(src)], llm, embed=lambda xs: [vecs[x] for x in xs], min_similarity=0.6)
    assert res["accepted"] == 0 and res["rejection_reasons"] == {"low_similarity": 1}
