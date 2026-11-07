from datetime import date

from fastapi.testclient import TestClient

import api.app as api
from agent.contracts import AgentName, AgentRequest, AgentResult, AgentStatus


class FakeStore:
    def get(self):
        return None


class FakeProgramStore:
    pass


class FakeHistoryStore:
    pass


def install_request_scoped_stores(monkeypatch):
    monkeypatch.setattr(api, "stores", lambda user: {
        "profile": FakeStore(),
        "programs": FakeProgramStore(),
        "history": FakeHistoryStore(),
        "traces": FakeStore(),
    })
    monkeypatch.setattr(api, "close", lambda stores: None)
    monkeypatch.setattr(api, "model", lambda: object())
    monkeypatch.setattr(api, "exercises", lambda: [])


def test_chat_endpoint_routes_fitness_through_authenticated_legacy_adapter(monkeypatch):
    install_request_scoped_stores(monkeypatch)
    seen = {}

    def handle(llm, message, exercises, profile, programs, today, history):
        seen.update(message=message, today=today)
        return {"status": "ok", "tool": "search_exercises", "data": {}, "reply": "Workout options."}

    monkeypatch.setattr("agent.dispatcher.handle_message", handle)
    app_client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = app_client.post("/v1/chat", json={"message": "Suggest a workout"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["reply"] == "Workout options."
    assert response.json()["tool"] == "search_exercises"
    assert response.json()["agent"] == "fitness"
    assert seen["message"] == "Suggest a workout"
    assert isinstance(seen["today"], date)


def test_chat_endpoint_routes_nutrition_to_nutrition_agent(monkeypatch):
    install_request_scoped_stores(monkeypatch)
    calls = []

    class StubNutritionAgent:
        name = AgentName.NUTRITION

        def __init__(self, llm):
            calls.append("created")

        def run(self, request):
            calls.append(request.user_id)
            return AgentResult(
                request_id=request.request_id,
                agent=AgentName.NUTRITION,
                status=AgentStatus.OK,
                message="Protein guidance grounded in [S1].",
            )

    monkeypatch.setattr("agent.nutrition.NutritionAgent", StubNutritionAgent)

    def legacy_must_not_run(*args, **kwargs):
        raise AssertionError("fitness dispatcher should not run for nutrition-only input")

    monkeypatch.setattr("agent.dispatcher.handle_message", legacy_must_not_run)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = client.post("/v1/chat", json={"message": "Explain protein in my diet"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["tool"] == "nutrition_agent"
    assert response.json()["agent"] == "nutrition"
    assert "[S1]" in response.json()["reply"]
    assert calls == ["created", "verified-user"]


def test_mixed_chat_invokes_both_agents_and_preserves_fitness_confirmation(monkeypatch):
    install_request_scoped_stores(monkeypatch)
    fitness_calls = []
    nutrition_calls = []

    def handle(*args, **kwargs):
        fitness_calls.append(True)
        return {
            "status": "confirm",
            "tool": "build_program",
            "data": {"pending": {"tool": "build_program", "plan": {"weeks": 4}}},
            "reply": "Confirm the workout plan.",
        }

    class StubNutritionAgent:
        name = AgentName.NUTRITION

        def __init__(self, llm):
            pass

        def run(self, request):
            nutrition_calls.append(True)
            return AgentResult(
                request_id=request.request_id,
                agent=AgentName.NUTRITION,
                status=AgentStatus.OK,
                message="Nutrition evidence [S1].",
            )

    monkeypatch.setattr("agent.dispatcher.handle_message", handle)
    monkeypatch.setattr("agent.nutrition.NutritionAgent", StubNutritionAgent)
    monkeypatch.setattr(api, "issue_confirm", lambda user, pending, marker: "signed-confirm-token")
    monkeypatch.setattr(api, "_marker", lambda programs: [None, None])

    result = api.run_chat("verified-user", "Create a workout and suggest meals", date.today())

    assert fitness_calls == [True]
    assert nutrition_calls == [True]
    assert result["status"] == "confirm"
    assert result["agent"] == "multi_agent"
    assert result["confirm_token"] == "signed-confirm-token"
    assert "Fitness:" in result["reply"] and "Nutrition:" in result["reply"]



def test_mixed_chat_does_not_hide_nutrition_unavailability(monkeypatch):
    install_request_scoped_stores(monkeypatch)

    def handle(*args, **kwargs):
        return {"status": "ok", "tool": "general_chat", "data": {}, "reply": "Progressive overload explained."}

    class UnavailableNutritionAgent:
        name = AgentName.NUTRITION

        def __init__(self, llm):
            pass

        def run(self, request):
            return AgentResult(
                request_id=request.request_id,
                agent=AgentName.NUTRITION,
                status=AgentStatus.UNAVAILABLE,
                message="No nutrition evidence is available.",
            )

    monkeypatch.setattr("agent.dispatcher.handle_message", handle)
    monkeypatch.setattr("agent.nutrition.NutritionAgent", UnavailableNutritionAgent)

    result = api.run_chat(
        "verified-user",
        "Explain progressive overload and how protein supports recovery",
        date.today(),
    )

    assert result["status"] == "unavailable"
    assert result["agent"] == "multi_agent"
    assert result["confirm_token"] is None
    assert "Fitness:" in result["reply"] and "Nutrition:" in result["reply"]


def test_fitness_concept_response_exposes_domain_agent_without_changing_legacy_tool(monkeypatch):
    install_request_scoped_stores(monkeypatch)

    def handle(*args, **kwargs):
        return {"status": "ok", "tool": "general_chat", "data": {}, "reply": "Progressive overload explained."}

    monkeypatch.setattr("agent.dispatcher.handle_message", handle)
    client = TestClient(api.app)
    api.app.dependency_overrides[api.current_user] = lambda: "verified-user"
    try:
        response = client.post("/v1/chat", json={"message": "What is progressive overload?"})
    finally:
        api.app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["agent"] == "fitness"
    assert response.json()["tool"] == "general_chat"
    assert response.json()["status"] == "ok"
