from cci.domain.schema import Contradiction, Preferences
from cci.eval.metrics import asr_normalize, cer_counts, extraction_prf, macro_f1, wer_counts
from cci.nlp.case import ReviewPolicy, build_case
from cci.nlp.rules import Extraction
from cci.speech.diarization import Turn, der


def test_selective_policy_reasons():
    ex = Extraction(
        fields={"policy_number": "12345678", "incident_date": "2026-10-01"},
        contradictions=[Contradiction(field="incident_date", values=["2026-10-01", "2026-10-02"])],
        preferences=Preferences(channel="sms"),
    )
    case = build_case("claim_auto", 0.4, ex, ReviewPolicy(min_intent_prob=0.6))
    assert case.decision == "review"
    assert set(case.review_reasons) == {"low_intent_confidence", "missing_required_fields", "contradiction"}
    assert "notify_via_sms" in case.next_actions and "request_missing_info" in case.next_actions
    assert "НЕ УКАЗАНО" in case.protocol


def test_injuries_route_to_priority_queue():
    ex = Extraction(
        fields={
            "policy_number": "1",
            "incident_date": "d",
            "incident_place": "p",
            "injuries": "yes",
            "other_party": "no",
        }
    )
    case = build_case("claim_auto", 0.99, ex, ReviewPolicy())
    assert case.queue == "claims_priority" and case.decision == "auto"


def test_fields_outside_intent_schema_are_dropped():
    case = build_case(
        "payment_status", 0.9, Extraction(fields={"claim_number": "123456", "address": "x"}), ReviewPolicy()
    )
    assert case.fields == {"claim_number": "123456"}


def test_wer_and_number_normalization():
    ref = asr_normalize("номер сто двадцать три")
    assert wer_counts(ref, asr_normalize("номер 123")) == (0, 2)
    assert wer_counts("a b c", "a x c") == (1, 3)
    assert cer_counts("abc", "abd") == (1, 3)


def test_extraction_prf_excludes_contradicted_fields():
    r = extraction_prf([{"a": "1", "b": "2"}], [{"a": "1", "b": "3"}], [{"b"}])
    assert r["precision"] == 1.0 and r["recall"] == 1.0


def test_macro_f1():
    assert macro_f1(["a", "a", "b"], ["a", "b", "b"]) == (2 / 3 + 2 / 3) / 2


def test_der_perfect_and_swapped_labels():
    ref = [Turn("operator", 0, 1), Turn("client", 1.5, 3)]
    assert der(ref, [Turn("spk1", 0, 1), Turn("spk0", 1.5, 3)])["der"] == 0.0
    # one hypothesis speaker for both: best map = client -> 1 s confusion + 0.5 s false alarm
    assert abs(der(ref, [Turn("spk0", 0, 3)])["der"] - (1.0 + 0.5) / 2.5) < 0.02


def test_cluster_bootstrap_is_wider_when_calls_share_templates():
    from cci.eval.metrics import bootstrap_ci, cluster_bootstrap_ci

    # 3 templates x 50 calls: a template is either always right or always wrong.
    groups = [f"t{i // 50}" for i in range(150)]
    correct = [1.0 if g in ("t0", "t1") else 0.0 for g in groups]
    call_lo, call_hi = bootstrap_ci(correct)
    cl_lo, cl_hi = cluster_bootstrap_ci(groups, lambda idx: sum(correct[i] for i in idx) / len(idx))
    assert cl_hi - cl_lo > 3 * (call_hi - call_lo)
    assert cl_lo <= 2 / 3 <= cl_hi
