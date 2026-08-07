from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from jarvis_brain.routes.world import proactive_event_loop
from app.api import world_routes
from jarvis_brain.app import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def reset_world_loop() -> None:
    """Keep world API tests isolated from each other."""
    proactive_event_loop.event_memory_manager.clear()
    proactive_event_loop.latest_suggestions = []


@pytest.fixture(autouse=True)
def mock_browser_open():
    with patch("app.tools.browser_driver.webbrowser.open", return_value=True):
        yield


def test_world_run_once_returns_200() -> None:
    response = client.post("/world/run-once")

    assert response.status_code == 200


def test_world_run_once_response_status_is_success() -> None:
    response = client.post(
        "/world/run-once",
        json={"context": {"interests": ["cybersecurity", "ai", "Jarvis project"]}},
    )

    assert response.json()["status"] == "success"


def test_world_run_once_returns_events_collected_greater_than_zero() -> None:
    response = client.post("/world/run-once")

    assert response.json()["events_collected"] > 0


def test_world_briefing_returns_200() -> None:
    response = client.get("/world/briefing")

    assert response.status_code == 200


def test_world_briefing_returns_expected_title() -> None:
    response = client.get("/world/briefing")

    assert response.json()["title"] == "Jarvis World Intelligence Briefing"


def test_world_briefing_use_llm_false_works() -> None:
    response = client.get("/world/briefing", params={"use_llm": False})

    assert response.status_code == 200
    assert response.json()["title"] == "Jarvis World Intelligence Briefing"


def test_world_briefing_use_llm_falls_back_when_disabled() -> None:
    response = client.get("/world/briefing", params={"use_llm": True})

    assert response.status_code == 200
    assert response.json()["metadata"]["llm_assisted"] is False


def test_world_briefing_use_llm_returns_assisted_metadata(monkeypatch) -> None:
    class FakeWorldEngine:
        def create_briefing(self, **kwargs):
            return {
                "summary": "Using mock world intelligence feeds, refined summary.",
                "priority_items": ["Mock cloud IAM advisory"],
                "alerts": [],
                "project_relevance": ["Jarvis security roadmap"],
                "suggested_next_steps": ["Review security controls."],
                "evidence_event_ids": [],
                "metadata": {
                    "llm_assisted": True,
                    "llm_assisted_world": True,
                    "world_model": "fake-world-model",
                },
            }

    monkeypatch.setattr(world_routes, "llm_assisted_world_engine", FakeWorldEngine())

    response = client.get("/world/briefing", params={"use_llm": True})

    body = response.json()
    assert response.status_code == 200
    assert body["metadata"]["llm_assisted_world"] is True
    assert body["metadata"]["world_model"] == "fake-world-model"
    assert "refined summary" in body["llm_summary"]


def test_world_events_returns_events_after_run_once() -> None:
    client.post("/world/run-once")

    response = client.get("/world/events")

    assert response.status_code == 200
    assert response.json()


def test_world_events_supports_category_filter() -> None:
    client.post("/world/run-once")

    response = client.get("/world/events", params={"category": "cybersecurity"})
    events = response.json()

    assert response.status_code == 200
    assert events
    assert all(event["category"] == "cybersecurity" for event in events)


def test_world_events_supports_severity_filter() -> None:
    client.post("/world/run-once")

    response = client.get("/world/events", params={"severity": "high"})
    events = response.json()

    assert response.status_code == 200
    assert events
    assert all(event["severity"] == "high" for event in events)


def test_world_events_supports_high_priority_only_filter() -> None:
    client.post("/world/run-once")

    response = client.get("/world/events", params={"high_priority_only": True})
    events = response.json()

    assert response.status_code == 200
    assert events
    assert all(
        event["severity"] in {"high", "critical"}
        or event["should_alert"] is True
        or event["relevance_score"] >= 0.8
        for event in events
    )


def test_world_suggestions_returns_200() -> None:
    response = client.get("/world/suggestions")

    assert response.status_code == 200
    assert response.json() == []


def test_world_alerts_returns_200() -> None:
    response = client.get("/world/alerts")

    assert response.status_code == 200
    assert response.json() == []


def test_existing_brain_think_still_works() -> None:
    response = client.post("/brain/think", json={"raw_input": "Hey Jarvis YouTube"})

    assert response.status_code == 200
    assert response.json()["message"] == "Opening YouTube now."


def test_existing_dev_state_still_works() -> None:
    response = client.get("/dev/state")

    assert response.status_code == 200
    assert response.json()["project"]


def test_no_real_api_or_llm_usage() -> None:
    response = client.post("/world/run-once")

    assert response.status_code == 200
    assert response.json()["status"] == "success"
    assert response.json()["events"][0]["source_name"].startswith("Mock")
