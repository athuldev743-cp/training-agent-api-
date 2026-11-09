from unittest.mock import MagicMock

from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus
from agent.nutrition import SYSTEM_PROMPT
from agent.orchestrator import MultiAgentOrchestrator
from rag.search import search_exercises
from rag.store import Exercise, safe_exercises
from state.profile import UserProfile


def test_nutrition_prompt_requires_conflict_disclosure_and_supported_calculations():
    assert "If retrieved sources conflict" in SYSTEM_PROMPT
    assert "inputs, units, and formula are supported" in SYSTEM_PROMPT
    assert "ask for missing inputs" in SYSTEM_PROMPT


def test_fitness_exercise_filter_excludes_user_limitation_stress_areas():
    exercises = [
        Exercise(
            id="knee-heavy", name="Deep Squat", muscles=["quads"], primary=["quads"],
            equipment=["bodyweight"], difficulty="beginner", movement="squat",
            stresses=["knee"],
        ),
        Exercise(
            id="knee-light", name="Glute Bridge", muscles=["glutes"], primary=["glutes"],
            equipment=["bodyweight"], difficulty="beginner", movement="bridge",
            stresses=["lower_back"],
        ),
    ]

    safe = safe_exercises(exercises, ["knee"])

    assert [item.id for item in safe] == ["knee-light"]


def test_exercise_search_respects_equipment_and_difficulty_constraints():
    exercises = [
        Exercise(
            id="bodyweight-push", name="Push Up", muscles=["chest"], primary=["chest"],
            equipment=["bodyweight"], difficulty="beginner", movement="push",
        ),
        Exercise(
            id="barbell-push", name="Bench Press", muscles=["chest"], primary=["chest"],
            equipment=["barbell", "bench"], difficulty="advanced", movement="push",
        ),
    ]

    found = search_exercises(
        exercises, muscle="chest", available_equipment=[], max_difficulty="beginner"
    )

    assert [item.id for item in found] == ["bodyweight-push"]


class _CaptureAgent:
    def __init__(self, name):
        self.name = name
        self.request = None

    def run(self, request):
        self.request = request
        return AgentResult(
            request_id=request.request_id,
            agent=self.name,
            status=AgentStatus.OK,
            message=f"{self.name.value} completed",
        )


def test_orchestrator_passes_same_profile_and_request_id_to_both_agents():
    fitness = _CaptureAgent(AgentName.FITNESS)
    nutrition = _CaptureAgent(AgentName.NUTRITION)
    graph = MultiAgentOrchestrator(fitness, nutrition)
    profile = UserProfile(name="Evaluation User", goal="general")

    result = graph.invoke(AgentRequest(
        user_id="evaluation-user",
        message="Create a workout and a meal plan",
        profile=profile,
    ))

    assert result.status is AgentStatus.OK
    assert fitness.request.profile == profile
    assert nutrition.request.profile == profile
    assert fitness.request.request_id == nutrition.request.request_id
