from fastapi.testclient import TestClient

from jarvis_brain.app import app


client = TestClient(app)


def test_agent_narration_status_returns_200() -> None:
    response = client.get("/agents/narration/status")

    assert response.status_code == 200
    data = response.json()
    assert data["enabled"] is True
    assert data["chain_of_thought_exposed"] is False
    assert data["llm_generated"] is False


def test_recent_narration_contains_no_secrets() -> None:
    client.post("/agents/lifecycle/reset-demo")
    client.post("/agents/lifecycle/demo")

    response = client.get("/agents/narration/recent")

    assert response.status_code == 200
    serialized = response.text
    assert "sk-" not in serialized
    assert "password=" not in serialized
