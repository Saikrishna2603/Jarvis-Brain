from fastapi.testclient import TestClient

from jarvis_brain.app import app


client = TestClient(app)


def test_lifecycle_status_endpoint_returns_200() -> None:
    response = client.get("/agents/lifecycle/status")

    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert response.json()["live_streaming"] is True
    assert response.json()["storage"] == "in_memory"
    assert response.json()["demo_supported"] is True


def test_lifecycle_snapshot_endpoint_returns_200() -> None:
    response = client.get("/agents/lifecycle/snapshot")

    assert response.status_code == 200
    assert "active_agents" in response.json()


def test_demo_creates_agents_and_events() -> None:
    response = client.post("/agents/lifecycle/demo")
    body = response.json()

    assert response.status_code == 200
    assert body["metadata"]["demo"] is True
    assert body["active_agents"] or body["completed_agents"]
    assert body["recent_events"]


def test_reset_clears_demo_data() -> None:
    client.post("/agents/lifecycle/demo")

    response = client.post("/agents/lifecycle/reset-demo")
    body = response.json()

    assert response.status_code == 200
    assert body["active_agents"] == []
    assert body["completed_agents"] == []
    assert body["failed_agents"] == []
    assert body["archived_agents"] == []


def test_agent_detail_endpoint_and_missing_agent() -> None:
    demo = client.post("/agents/lifecycle/demo").json()
    agent_id = demo["active_agents"][0]["agent_id"]

    response = client.get(f"/agents/lifecycle/{agent_id}")
    missing = client.get("/agents/lifecycle/agent-does-not-exist")

    assert response.status_code == 200
    assert response.json()["agent"]["agent_id"] == agent_id
    assert response.json()["events"]
    assert missing.status_code == 404


def test_lifecycle_api_does_not_expose_secrets() -> None:
    client.post("/agents/lifecycle/demo")
    response = client.get("/agents/lifecycle/snapshot")
    serialized = str(response.json())

    assert "sk-test" not in serialized
    assert "password=" not in serialized
