from fastapi.testclient import TestClient

from api.app import app
import evaluation.resume_metrics as metrics

SECRET = "test-evaluation-secret-that-is-at-least-32-chars"
STATIC_PATH = "/internal/evaluation/resume-metrics/static"
NUTRITION_PATH = "/internal/evaluation/resume-metrics/nutrition/0"


def test_resume_evaluation_routes_are_hidden_from_openapi():
    paths = app.openapi()["paths"]
    assert STATIC_PATH not in paths
    assert NUTRITION_PATH not in paths


def test_resume_evaluation_requires_configured_secret(monkeypatch):
    monkeypatch.delenv("EVALUATION_API_KEY", raising=False)
    with TestClient(app) as client:
        response = client.post(STATIC_PATH)
    assert response.status_code == 404


def test_static_resume_evaluation_is_secret_gated_and_read_only(monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)
    monkeypatch.setattr(
        metrics,
        "run_static_resume_evaluations",
        lambda: {
            "evaluation": "live-resume-metrics-static",
            "routing": {"query_count": 40, "accuracy": 0.875},
            "workout_constraints": {"profiles_total": 90},
            "progression": {"total": 14, "passed": 14},
            "safety": {"total": 35, "passed": 34},
            "user_data_mutated": False,
        },
    )
    with TestClient(app) as client:
        response = client.post(STATIC_PATH, headers={"X-Evaluation-Key": SECRET})
    assert response.status_code == 200
    body = response.json()
    assert body["routing"]["accuracy"] == 0.875
    assert body["progression"] == {"total": 14, "passed": 14}
    assert body["user_data_mutated"] is False
    assert body["nutrition_grounding_dataset_cases"] > 0


def test_nutrition_case_endpoint_returns_only_evaluation_result(monkeypatch):
    monkeypatch.setenv("EVALUATION_API_KEY", SECRET)
    monkeypatch.setattr(
        routes,
        "require_evaluation_key",
        lambda: None,
    )
    monkeypatch.setattr(
        metrics,
        "evaluate_nutrition_case",
        lambda index: {
            "case_id": "nutrition-001",
            "case_index": index,
            "status": "ok",
            "citation_contract_valid": True,
            "grounded_answer_success": True,
            "answer": "Test answer [S1]",
        },
    )
    with TestClient(app) as client:
        response = client.post(NUTRITION_PATH, headers={"X-Evaluation-Key": SECRET})
    assert response.status_code == 200
    assert response.json()["citation_contract_valid"] is True


def test_progression_dataset_reproduces_fourteen_cases():
    report = metrics.evaluate_progression()
    assert report["total"] == 14
    assert report["passed"] == 14
    assert report["pass_rate"] == 1.0


def test_nutrition_aggregate_keeps_human_review_separate():
    report = metrics.aggregate_nutrition_cases([
        {"citation_contract_valid": True, "insufficient_evidence_refusal": False},
        {"citation_contract_valid": False, "insufficient_evidence_refusal": True},
    ])
    assert report["grounded_answer_successes"] == 1
    assert report["total_requests"] == 2
    assert report["hand_checked_citation_validity"] is None
    assert report["manual_factual_support_review_required"] is True
