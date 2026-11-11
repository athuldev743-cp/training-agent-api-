from unittest.mock import MagicMock

from agent.contracts import AgentRequest, AgentStatus
from agent.nutrition import NutritionAgent, _expand_grouped_citations, _is_insufficient_evidence


def _agent(text):
    llm = MagicMock()
    llm.create_chat_completion.return_value = {"choices": [{"message": {"content": text}}]}
    chunk = {"chunk_id": "a" * 64, "content": "Salt under 5 g per day.", "source": "who.txt", "page": 1, "chunk_number": 0}
    return NutritionAgent(llm, retriever=lambda *a, **k: [chunk] * 3)


def _run(text):
    return _agent(text).run(AgentRequest(user_id="u", message="How much salt per day"))


def test_grouped_labels_are_expanded():
    assert _expand_grouped_citations("Limit salt [S1, S3].") == "Limit salt [S1][S3]."


def test_grouped_valid_citations_pass():
    r = _run("Limit salt to 5 g [S1, S3].")
    assert r.status is AgentStatus.OK and "[S1][S3]" in r.message


def test_grouped_unknown_label_still_fails_closed():
    r = _run("Limit salt to 5 g [S1, S9].")
    assert r.status is AgentStatus.UNAVAILABLE


def test_uncited_claim_with_numbers_still_fails_closed():
    r = _run("Limit salt to 5 g per day.")
    assert r.status is AgentStatus.UNAVAILABLE


def test_honest_refusal_is_accepted_without_evidence():
    r = _run("The available sources do not contain enough information to answer that.")
    assert r.status is AgentStatus.REFUSED and r.evidence == []


def test_refusal_helper_rejects_numbers_and_long_text():
    assert not _is_insufficient_evidence("The sources do not contain info but eat 2000 kcal.")
    assert not _is_insufficient_evidence("do not contain " + "x" * 500)
