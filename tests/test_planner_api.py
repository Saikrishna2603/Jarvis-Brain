from fastapi.testclient import TestClient

from jarvis_brain.routes.brain import brain_engine
from jarvis_brain.app import app


client = TestClient(app)


def test_planner_preview_returns_200() -> None:
    response = client.post(
        "/planner/preview",
        json={"raw_input": "review my finances"},
    )

    assert response.status_code == 200
    assert len(response.json()["steps"]) == 3


def test_planner_preview_use_llm_false_returns_rule_plan() -> None:
    response = client.post(
        "/planner/preview",
        json={"raw_input": "secure my home", "use_llm": False},
    )

    assert response.status_code == 200
    assert response.json()["metadata"] == {}
    assert response.json()["steps"][0]["action"] == "lock_door"


def test_planner_preview_use_llm_falls_back_when_disabled() -> None:
    original_enabled = brain_engine.llm_assisted_planner.enabled
    brain_engine.llm_assisted_planner.enabled = False
    try:
        response = client.post(
            "/planner/preview",
            json={
                "raw_input": "review my finances",
                "use_llm": True,
            },
        )
    finally:
        brain_engine.llm_assisted_planner.enabled = original_enabled

    assert response.status_code == 200
    assert response.json()["steps"]
    assert response.json()["metadata"] == {}


def test_planner_preview_does_not_execute_or_create_tasks() -> None:
    brain_engine.task_memory_manager.clear()
    before_plans = len(brain_engine.plan_memory_manager.get_all_plans())

    response = client.post(
        "/planner/preview",
        json={"raw_input": "review my finances", "use_llm": True},
    )

    assert response.status_code == 200
    assert brain_engine.task_memory_manager.get_all_tasks() == []
    assert len(brain_engine.plan_memory_manager.get_all_plans()) == before_plans
