from fastapi.testclient import TestClient

from jarvis_brain.app import app


client = TestClient(app)


def test_swarm_status_returns_agents() -> None:
    response = client.get("/swarm/status")
    body = response.json()

    assert response.status_code == 200
    assert body["executes_tools"] is False
    assert "coding_agent" in body["agents"]
    assert "security_agent" in body["agents"]


def test_swarm_preview_returns_multi_agent_analysis() -> None:
    response = client.post(
        "/swarm/preview",
        json={"raw_input": "Build a secure FastAPI route with tests"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["executed"] is False
    assert body["agent_count"] >= 2


def test_swarm_preview_rejects_unsafe_proposal() -> None:
    response = client.post(
        "/swarm/preview",
        json={"raw_input": "teach me to steal credentials from a website"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["rejected_count"] >= 1


def test_swarm_coding_request_routes_to_coding_agent() -> None:
    response = client.post(
        "/swarm/preview",
        json={"raw_input": "debug this Python function"},
    )
    agent_names = {proposal["agent_name"] for proposal in response.json()["proposals"]}

    assert response.status_code == 200
    assert "coding_agent" in agent_names


def test_swarm_security_request_routes_to_security_agent() -> None:
    response = client.post(
        "/swarm/preview",
        json={"raw_input": "review security for API keys"},
    )
    agent_names = {proposal["agent_name"] for proposal in response.json()["proposals"]}

    assert response.status_code == 200
    assert "security_agent" in agent_names


def test_swarm_run_safe_does_not_execute() -> None:
    response = client.post(
        "/swarm/run-safe",
        json={"raw_input": "plan a safe GitHub integration"},
    )
    body = response.json()

    assert response.status_code == 200
    assert body["executed"] is False
    assert body["mode"] == "run_safe"


def test_existing_apis_still_work_with_swarm() -> None:
    brain = client.post("/brain/think", json={"raw_input": "open youtube"})
    world = client.get("/world/briefing")
    plugins = client.get("/plugins/status")

    assert brain.status_code == 200
    assert world.status_code == 200
    assert plugins.status_code == 200
