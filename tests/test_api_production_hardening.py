import logging

from fastapi.testclient import TestClient

import api.app as api


def _providers(monkeypatch):
    from agent import cloud_llm
    monkeypatch.setattr(
        cloud_llm, "PROVIDERS",
        [("test-provider", None, "TEST_PROVIDER_KEY", None, None)],
    )
    monkeypatch.setenv("TEST_PROVIDER_KEY", "configured")


def test_request_id_and_latency_are_logged_without_request_body(caplog):
    caplog.set_level(logging.INFO, logger="training_agent.api")
    with TestClient(api.app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert len(response.headers["X-Request-ID"]) == 32
    assert "http_request_completed" in caplog.text
    assert any(
        getattr(record, "http_path", None) == "/health"
        and hasattr(record, "duration_ms")
        for record in caplog.records
    )


def test_ready_returns_503_when_postgres_is_unavailable(monkeypatch):
    monkeypatch.setenv("STATE_BACKEND", "postgres")
    monkeypatch.setenv("API_TOKEN_SECRET", "x" * 40)
    _providers(monkeypatch)

    def fail():
        raise RuntimeError("private database connection string")

    monkeypatch.setattr("rag.neon.check_connection", fail)
    response = TestClient(api.app).get("/ready")

    assert response.status_code == 503
    assert response.json()["ready"] is False
    assert response.json()["database_ready"] is False
    assert "private database connection string" not in response.text


def test_ready_returns_200_when_postgres_and_required_config_are_available(monkeypatch):
    monkeypatch.setenv("STATE_BACKEND", "postgres")
    monkeypatch.setenv("API_TOKEN_SECRET", "x" * 40)
    _providers(monkeypatch)
    monkeypatch.setattr(
        "rag.neon.check_connection",
        lambda: {"status": "connected", "database": "redacted", "user": "redacted"},
    )

    response = TestClient(api.app).get("/ready")

    assert response.status_code == 200
    assert response.json()["database_ready"] is True
    assert "redacted" not in response.text


def test_rate_limit_is_enforced_but_health_remains_exempt(monkeypatch):
    monkeypatch.setitem(api.RATE, "limit", 1)
    api._hits.clear()
    try:
        with TestClient(api.app) as client:
            first = client.get("/v1/profile")
            second = client.get("/v1/profile")
            health = client.get("/health")
    finally:
        api._hits.clear()

    assert first.status_code == 401
    assert second.status_code == 429
    assert health.status_code == 200


def test_shutdown_closes_postgres_worker_connections(monkeypatch):
    calls = []
    monkeypatch.setenv("STATE_BACKEND", "postgres")
    monkeypatch.setattr("state.pg_compat.close_all", lambda: calls.append("closed"))

    api.close_database_connections()

    assert calls == ["closed"]
