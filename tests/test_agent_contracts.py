from datetime import date
from uuid import uuid4

import pytest
from pydantic import ValidationError

from agent.contracts import (
    AgentName,
    AgentRequest,
    AgentResult,
    AgentStatus,
    EvidenceReference,
)
from agent.interfaces import DomainAgent


def test_agent_request_applies_defaults_and_preserves_profile():
    request = AgentRequest(
        user_id="user-123",
        message="Build a beginner workout",
        today=date(2026, 11, 8),
        profile={
            "name": "Test User",
            "goal": "strength",
            "experience": "beginner",
            "limitations": ["knee"],
        },
    )

    assert request.request_id
    assert request.profile is not None
    assert request.profile.goal == "strength"
    assert request.profile.limitations == ["knee"]


def test_agent_request_rejects_empty_or_oversized_input():
    with pytest.raises(ValidationError):
        AgentRequest(user_id="user-123", message=" ")

    with pytest.raises(ValidationError):
        AgentRequest(user_id="user-123", message="x" * 1001)


def test_agent_request_rejects_unknown_top_level_fields():
    with pytest.raises(ValidationError):
        AgentRequest(user_id="user-123", message="Hello", unexpected="value")


def test_evidence_reference_validates_source_and_page():
    assert EvidenceReference(source="Guideline", page=2).page == 2

    with pytest.raises(ValidationError):
        EvidenceReference(source="", page=0)


def test_agent_result_requires_correlated_request_and_valid_status():
    request_id = uuid4()
    result = AgentResult(
        request_id=request_id,
        agent=AgentName.NUTRITION,
        status=AgentStatus.OK,
        message="Use a balanced meal pattern.",
        evidence=[EvidenceReference(source="Nutrition guideline", page=4)],
    )

    assert result.request_id == request_id
    assert result.agent is AgentName.NUTRITION
    assert len(result.evidence) == 1


def test_agent_result_rejects_unknown_fields_and_empty_message():
    common = {
        "request_id": uuid4(),
        "agent": AgentName.FITNESS,
        "status": AgentStatus.OK,
    }

    with pytest.raises(ValidationError):
        AgentResult(**common, message=" ", extra_output="not allowed")


def test_domain_agent_protocol_is_structural():
    class FitnessStub:
        name = AgentName.FITNESS

        def run(self, request: AgentRequest) -> AgentResult:
            return AgentResult(
                request_id=request.request_id,
                agent=self.name,
                status=AgentStatus.OK,
                message="Workout plan ready.",
            )

    assert isinstance(FitnessStub(), DomainAgent)
