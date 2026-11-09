from unittest.mock import MagicMock

import pytest

from agent.contracts import AgentName, AgentRequest, AgentStatus
from agent.interfaces import DomainAgent
from agent.nutrition import NutritionAgent
from state.profile import UserProfile


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



def test_saved_profile_is_included_in_nutrition_prompt_and_constraints():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Choose protein-rich vegetarian meals [S1]."}}]
    }
    profile = UserProfile(
        name="Test User", age=29, height_cm=172, weight_kg=78,
        goal="fat_loss", activity_level="moderately_active",
        dietary_pattern="vegetarian", preferred_foods=["dal", "paneer"],
        excluded_foods=["mushroom"], allergies=["peanut"],
        intolerances=["lactose"], meals_per_day=4, daily_food_budget_inr=250,
    )
    agent = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()])
    result = agent.run(AgentRequest(
        user_id="test-user", message="Create a practical meal plan",
        profile=profile,
    ))

    assert result.status is AgentStatus.OK
    system = llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]
    for value in ('"goal": "fat_loss"', '"activity_level": "moderately_active"',
                  '"dietary_pattern": "vegetarian"', '"preferred_foods": ["dal", "paneer"]',
                  '"excluded_foods": ["mushroom"]', '"allergies": ["peanut"]',
                  '"intolerances": ["lactose"]', '"meals_per_day": 4',
                  '"daily_food_budget_inr": 250'):
        assert value in system
    assert "Do not assume personal details" not in system


def test_missing_profile_is_explicit_and_not_inferred():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "The source discusses balanced meals [S1]."}}]
    }
    result = NutritionAgent(llm, retriever=lambda *a, **k: [chunk()]).run(request())
    system = llm.create_chat_completion.call_args.kwargs["messages"][0]["content"]

    assert result.status is AgentStatus.OK
    assert "No saved profile is available" in system
    assert "Do not assume personal details" in system



def test_personalized_request_uses_profile_terms_for_hybrid_retrieval():
    llm = MagicMock()
    llm.create_chat_completion.return_value = {
        "choices": [{"message": {"content": "Build meals around these preferences [S1]."}}]
    }
    calls = []
    profile = UserProfile(
        name="Test User", goal="fat_loss", activity_level="moderately_active",
        dietary_pattern="vegetarian", preferred_foods=["dal"],
        excluded_foods=["mushroom"],
    )

    def retriever(query, **kwargs):
        calls.append(query)
        return [chunk()]

    result = NutritionAgent(llm, retriever=retriever).run(AgentRequest(
        user_id="test-user", message="Create a meal plan", profile=profile,
    ))

    assert result.status is AgentStatus.OK
    assert "fat_loss" in calls[0]
    assert "vegetarian" in calls[0]
    assert "dal" in calls[0]
    assert "mushroom" in calls[0]
