from unittest.mock import MagicMock

import pytest

from agent.contracts import AgentName, AgentRequest, AgentStatus
from agent.interfaces import DomainAgent
from agent.nutrition import NutritionAgent


def request(message="How does protein support muscle recovery?"):
    return AgentRequest(user_id="test-user", message=message)


def chunk(source="sports_nutrition.pdf", page=4, text="Dietary protein supports muscle protein synthesis."):
    return {
        "chunk_id": "a" * 64,
        "content": text,
        "source_id": "sports-guide",
        "source": source,
        "title": "Sports Nutrition",
        "page": page,
        "chunk_number": 2,
        "rrf_score": 0.02,
    }


def test_nutrition_agent_implements_domain_protocol():
    agent = NutritionAgent(MagicMock(), retriever=lambda *a, **k: [chunk()])
    assert isinstance(agent, DomainAgent)
    assert agent.name is AgentName.NUTRITION


def test_returns_cited_answer_with_source_attribution():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Protein supports muscle protein synthesis [S1]."}}]
    }
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()])

    result = agent.run(request())

    assert result.status is AgentStatus.OK
    assert result.agent is AgentName.NUTRITION
    assert result.evidence[0].source == "sports_nutrition.pdf"
    assert result.evidence[0].page == 4
    assert "[S1]" in result.message
    system = llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]
    assert "RETRIEVED EVIDENCE" in system


def test_empty_retrieval_does_not_call_llm():
    llm = MagicMock()
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [])

    result = agent.run(request())

    assert result.status is AgentStatus.UNAVAILABLE
    llm.create_chat_completion.assert_not_called()


def test_retrieval_failure_returns_controlled_unavailable():
    def fail(*args, **kwargs):
        raise RuntimeError("database credentials must not leak")

    result = NutritionAgent(MagicMock(), retriever=fail).run(request())

    assert result.status is AgentStatus.UNAVAILABLE
    assert "credentials" not in result.message


def test_invalid_model_citation_fails_closed():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Take this supplement [S9]."}}]
    }
    result = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]).run(request())

    assert result.status is AgentStatus.UNAVAILABLE
    assert result.evidence[0].source == "sports_nutrition.pdf"


def test_uncited_model_response_fails_closed():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Protein is important for recovery."}}]
    }
    result = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]).run(request())

    assert result.status is AgentStatus.UNAVAILABLE


def test_model_failure_returns_evidence_and_controlled_message():
    llm = MagicMock()
    llm.create_chat_completion.side_effect = RuntimeError("provider error")
    result = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]).run(request())

    assert result.status is AgentStatus.UNAVAILABLE
    assert len(result.evidence) == 1
    assert "provider error" not in result.message


def test_empty_retrieved_chunk_is_not_used_as_evidence():
    llm = MagicMock()
    result = NutritionAgent(
        llm, retriever=lambda *a, **k: [chunk(text="  ")]
    ).run(request())

    assert result.status is AgentStatus.UNAVAILABLE
    llm.create_chat_completion.assert_not_called()


def test_retrieval_call_is_bounded_and_reranked():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Protein supports muscle protein synthesis [S1]."}}]
    }
    calls = []
    def retriever(*args, **kwargs):
        calls.append((args, kwargs))
        return [chunk()]

    NutritionAgent(llm, retriever=retriever).run(request())
    assert calls[0][1] == {"limit": 5, "candidate_limit": 30, "rerank": True}


def test_agent_rejects_invalid_evidence_limit():
    with pytest.raises(ValueError):
        NutritionAgent(MagicMock(), max_evidence=6)


def test_citation_labels_skip_empty_chunks_without_gaps():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Protein supports recovery [S1]."}}]
    }
    result = NutritionAgent(
        llm,
        retriever=lambda *a, **k: [chunk(text=" "), chunk(source="second.pdf")],
    ).run(request())

    assert result.status is AgentStatus.OK
    assert result.evidence[0].source == "second.pdf"

def test_medical_nutrition_request_is_blocked_before_retrieval_or_llm():
    llm = MagicMock()
    retrieval_calls = []
    def retriever(*args, **kwargs):
        retrieval_calls.append(True)
        return [chunk()]

    result = NutritionAgent(llm, retriever=retriever).run(
        request("Create a meal plan to manage diabetes")
    )

    assert result.status is AgentStatus.REFUSED
    assert retrieval_calls == []
    llm.create_chat_completion.assert_not_called()


def test_general_condition_education_is_not_blocked():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Diabetes is a metabolic condition [S1]."}}]
    }
    result = NutritionAgent(
        llm, retriever=lambda *a, **k: [chunk()]
    ).run(request("What is diabetes?"))

    assert result.status is AgentStatus.OK
    llm.create_chat_completion.assert_called_once()
