"""Offline tests: no API key or network needed.

The model is replaced by a scripted fake so the verification and retry logic
can be tested against deliberately bad output (paraphrased quotes, wrong
timestamps, invented numbers, interviewer quotes, wrong expert).
"""

import pytest

from core import llm
from core.generation import check_claim_numbers, numbers_in
from core.pipeline import Pipeline
from core.verify import CitationVerifier, VerifiedCitation, normalize, verify_payload


@pytest.fixture(scope="module")
def pipe():
    return Pipeline()


# --- parser -------------------------------------------------------------------

def test_parser_loads_three_experts(pipe):
    names = [t.expert_name for t in pipe.transcripts]
    assert names == ["Dr. Jean Martin", "Anna Keller", "Dr. Emily Carter"]
    assert [t.market for t in pipe.transcripts] == ["France", "Germany", "United Kingdom"]
    for t in pipe.transcripts:
        assert len(t.segments) == 14 and len(t.expert_segments) == 7
    assert len(pipe.questions) == 6


def test_parser_keeps_text_verbatim(pipe):
    seg = pipe.verifier.segments["DE@06:05"]
    assert seg.text == (
        "Nine to eighteen months is common. Procurement, clinical leadership, finance "
        "and management all need to align, so it can move slowly."
    )
    assert seg.timestamp_seconds == 365 and seg.speaker == "Anna Keller"


# --- verifier -----------------------------------------------------------------

def test_exact_quote_is_resliced_from_source(pipe):
    r = pipe.verifier.verify("FR@05:07", "I WOULD expect maybe 15 to 20 percent more procedures annually,")
    assert isinstance(r, VerifiedCitation)
    assert r.quote == "I would expect maybe 15 to 20 percent more procedures annually"
    assert r.timestamp == "05:07" and r.status == "exact"


@pytest.mark.parametrize(
    "seg_id, quote",
    [
        ("FR@05:07", "I would expect about 15 to 20 percent more procedures annually"),  # paraphrase
        ("FR@05:07", "I expect adoption ... smaller hospitals will remain slower"),     # ellipsis
        ("FR@05:00", "What do you expect over the next three to five years"),          # interviewer
        ("FR@05:07", "steadily"),                                                      # too short
        ("FR@99:99", "I expect adoption to continue increasing"),                      # bad id
        ("UK@04:06", "I expect adoption to continue increasing"),                      # wrong expert text
        ("FR@00:18", "I expect adoption to continue increasing"),                      # far from source
    ],
)
def test_bad_quotes_rejected(pipe, seg_id, quote):
    assert not isinstance(pipe.verifier.verify(seg_id, quote), VerifiedCitation)


def test_word_boundary(pipe):
    # a truncated word ("becom" for "become") must not match as a prefix
    assert not isinstance(pipe.verifier.verify("FR@03:10", "the economics becom difficult"), VerifiedCitation)


def test_adjacent_turn_is_corrected_not_trusted(pipe):
    r = pipe.verifier.verify("FR@04:08", "I expect adoption to continue increasing")
    assert isinstance(r, VerifiedCitation)
    assert r.segment_id == "FR@05:07" and r.timestamp == "05:07" and r.status == "corrected"


def test_scope_restricts_expert(pipe):
    r = pipe.verifier.verify("DE@06:05", "Nine to eighteen months is common", {"FR"})
    assert not isinstance(r, VerifiedCitation)


def test_position_expert_id_enforced(pipe):
    payload = {"positions": [{"expert_id": "UK", "position": "x",
                              "citations": [{"segment_id": "DE@06:05", "quote": "Nine to eighteen months is common"}]}]}
    out, failures = verify_payload(payload, pipe.verifier)
    assert out["positions"][0]["citations"] == [] and len(failures) == 1


def test_normalize_only_case_space_punct():
    assert normalize("  It’s   GROWING, but—slowly. ") == "it s growing but slowly"


# --- claim number check -------------------------------------------------------

def test_numbers_in_maps_words():
    assert numbers_in("Six to twelve months") == {"6", "12"}


def test_claim_with_invented_number_is_rejected():
    payload = {"claim": "Dr. Martin expects 25 percent growth",
               "citations": [{"quote": "I would expect maybe 15 to 20 percent more procedures annually"}]}
    issues = check_claim_numbers(payload)
    assert issues and payload["citations"] == []


def test_question_numbers_and_expert_count_allowed():
    payload = {"claim": "All three experts expect growth over the next 3-5 years",
               "citations": [{"quote": "I expect adoption to continue increasing"}]}
    assert check_claim_numbers(payload, frozenset({"3", "5"})) == [] and payload["citations"]


def test_claim_with_matching_numbers_passes():
    payload = {"claim": "Six to twelve months is realistic",
               "citations": [{"quote": "Six to twelve months is realistic once the hospital becomes serious"}]}
    assert check_claim_numbers(payload) == [] and payload["citations"]


# --- retrieval ----------------------------------------------------------------

@pytest.mark.parametrize(
    "q_index, expert, expected",
    [
        (0, "FR", "FR@00:18"), (1, "DE", "DE@01:10"), (2, "DE", "DE@02:08"),
        (3, "DE", "DE@03:05"), (4, "FR", "FR@05:07"), (5, "UK", "UK@05:04"),
        (5, "FR", "FR@06:08"), (5, "DE", "DE@06:05"),
    ],
)
def test_retrieval_top_hit(pipe, q_index, expert, expected):
    hits = pipe.retriever.search(pipe.questions[q_index], [expert], top_k=5)
    assert expected in [h.segment.segment_id for h in hits[:2]]


def test_retrieval_returns_nothing_for_unrelated_question(pipe):
    assert pipe.retriever.search_per_expert("What is the weather in Paris?", pipe.expert_ids, 3) == []


# --- generation loop with a scripted fake model --------------------------------

class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def __call__(self, system, messages, schema, **kw):
        self.calls.append(messages)
        return self.responses.pop(0)


def test_retry_fixes_bad_quote_and_drops_unfixable(monkeypatch, pipe):
    first = {"answerable": True, "not_covered": "", "points": [
        {"claim": "Purchases take six to twelve months.",
         "citations": [{"segment_id": "FR@06:08", "quote": "Six to 12 months is realistic"}]},
        {"claim": "Dr. Martin says hospitals buy quickly.",
         "citations": [{"segment_id": "FR@06:08", "quote": "hospitals buy quickly"}]},
    ]}
    second = {"answerable": True, "not_covered": "", "points": [
        {"claim": "Purchases take six to twelve months.",
         "citations": [{"segment_id": "FR@06:08", "quote": "Six to twelve months is realistic"}]},
        {"claim": "Dr. Martin says hospitals buy quickly.",
         "citations": [{"segment_id": "FR@06:08", "quote": "hospitals buy quickly"}]},
    ]}
    fake = FakeLLM([first, second])
    monkeypatch.setattr(llm, "call_json", fake)
    ans = pipe.answer_guide_question(pipe.questions[5], "FR", use_cache=False)
    assert len(fake.calls) == 2
    assert "exact text of FR@06:08" in fake.calls[1][-1]["content"]  # source text fed back
    assert ans.status == "answered" and ans.attempts == 2
    assert [p["claim"] for p in ans.points] == ["Purchases take six to twelve months."]
    assert ans.points[0]["citations"][0]["quote"] == "Six to twelve months is realistic"
    assert len(ans.failures_first_pass) == 2 and len(ans.failures_final) == 1


def test_not_answerable_is_passed_through(monkeypatch, pipe):
    fake = FakeLLM([{"answerable": False, "points": [], "not_covered": "Japan is not discussed."}])
    monkeypatch.setattr(llm, "call_json", fake)
    ans = pipe.ask("What about adoption in Japan?", use_cache=False)
    assert ans.status == "not_discussed" and ans.points == []


def test_no_retrieval_hits_skips_model(monkeypatch, pipe):
    fake = FakeLLM([])
    monkeypatch.setattr(llm, "call_json", fake)
    ans = pipe.ask("What is the weather in Paris?", use_cache=False)
    assert ans.status == "not_discussed" and fake.calls == []


def test_all_quotes_fabricated_means_no_answer(monkeypatch, pipe):
    bad = {"answerable": True, "not_covered": "", "points": [
        {"claim": "Hospitals love robots.",
         "citations": [{"segment_id": "UK@00:14", "quote": "hospitals love robots a lot"}]}]}
    fake = FakeLLM([bad, bad])
    monkeypatch.setattr(llm, "call_json", fake)
    ans = pipe.answer_guide_question(pipe.questions[0], "UK", use_cache=False)
    assert ans.status == "no_verified_support" and ans.points == []
