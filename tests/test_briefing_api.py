import pytest
from fastapi.testclient import TestClient

from jarvis_brain.briefing.briefing_dependencies import briefing_store, skill_registry
from jarvis_brain.engine.intent_resolver import IntentResolver
from jarvis_brain.app import app


client = TestClient(app)


@pytest.fixture(autouse=True)
def clean_briefing_store():
    """Keep briefing history isolated between tests."""
    briefing_store.clear()
    yield
    briefing_store.clear()


# --------------------------------------------------------------------------
# "Hey Jarvis, good morning."
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "phrase",
    [
        "Hey Jarvis, good morning.",
        "good morning",
        "Good Morning, Jarvis!",
        "morning jarvis",
        "brief me",
        "what did I miss?",
    ],
)
def test_good_morning_resolves_to_the_briefing_intent(phrase: str) -> None:
    """The primary trigger works through the rule-based resolver."""
    intent = IntentResolver().resolve(phrase)

    assert intent.intent_type == "daily_briefing"
    assert intent.action == "daily_briefing"


def test_good_morning_command_returns_a_briefing() -> None:
    """The explicit command produces the real briefing."""
    response = client.post(
        "/briefing/command",
        json={"text": "Hey Jarvis, good morning."},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["matched"] is True
    assert body["intent"] == "daily_briefing"
    assert body["briefing"]["spoken_summary"]
    assert body["briefing"]["greeting"].startswith("Good ")


def test_unrelated_command_does_not_trigger_a_briefing() -> None:
    """Only the briefing intent produces a briefing."""
    response = client.post("/briefing/command", json={"text": "open youtube"})

    body = response.json()
    assert body["matched"] is False
    assert body["briefing"] is None


# --------------------------------------------------------------------------
# The briefing endpoint.
# --------------------------------------------------------------------------


def test_daily_briefing_is_generated_from_real_sources() -> None:
    """The endpoint returns a structured, attributable briefing."""
    briefing = client.get("/briefing/daily").json()

    assert briefing["generated_from_real_data"] is True
    assert briefing["demo"] is False
    assert briefing["sections"]
    assert briefing["spoken_summary"]

    for section in briefing["sections"]:
        if section["available"]:
            assert section["summary"]
        else:
            assert section["unavailable_reason"]


def test_unconfigured_sources_are_reported_not_hidden() -> None:
    """Messages, calendar, and world are honestly unavailable in this system."""
    briefing = client.get("/briefing/daily").json()

    sections = {section["type"]: section for section in briefing["sections"]}

    assert sections["messages"]["available"] is False
    assert sections["schedule"]["unavailable_reason"] == "Calendar access is not configured."
    assert sections["world"]["available"] is False
    assert "No trusted world intelligence source is configured." in sections["world"]["summary"]

    assert briefing["unavailable_sources"]
    assert briefing["partial"] is True


def test_refresh_produces_a_new_briefing() -> None:
    """Refresh re-collects every source."""
    first = client.get("/briefing/daily").json()
    second = client.post("/briefing/refresh").json()

    assert second["briefing_id"] != first["briefing_id"]
    assert second["generated_at"] >= first["generated_at"]


def test_replay_records_without_regenerating() -> None:
    """Replay re-speaks what was already said; it does not re-collect."""
    briefing = client.get("/briefing/daily").json()
    briefing_id = briefing["briefing_id"]

    response = client.post(f"/briefing/{briefing_id}/replay")

    assert response.status_code == 200
    record = response.json()
    assert record["briefing_id"] == briefing_id
    assert record["replayed_count"] == 1


def test_dismiss_and_spoken_are_recorded() -> None:
    """The store remembers that a briefing was spoken and dismissed."""
    briefing_id = client.get("/briefing/daily").json()["briefing_id"]

    client.post(f"/briefing/{briefing_id}/spoken")
    record = client.post(f"/briefing/{briefing_id}/dismiss").json()

    assert record["spoken"] is True
    assert record["dismissed"] is True


def test_replay_of_unknown_briefing_is_404() -> None:
    """An unknown briefing id is an honest 404."""
    assert client.post("/briefing/does-not-exist/replay").status_code == 404


def test_history_stores_only_safe_metadata() -> None:
    """Persisted records carry identifiers, never content."""
    client.get("/briefing/daily")

    history = client.get("/briefing/history").json()

    assert len(history) == 1
    record = history[0]
    assert record["briefing_id"]
    assert "section_availability" in record

    # Nothing that could carry private content may be persisted.
    for forbidden in ("spoken_summary", "greeting", "sections", "items", "body", "audio"):
        assert forbidden not in record


# --------------------------------------------------------------------------
# Skills.
# --------------------------------------------------------------------------


def test_skills_status_declares_its_guarantees() -> None:
    """The skill subsystem states plainly what it will not do."""
    status = client.get("/skills/status").json()

    assert status["llm_may_recommend_skills"] is False
    assert status["autonomous_installation_enabled"] is False
    assert status["research_and_installation_approvals_separate"] is True
    assert status["recommendations_source"] == "reviewed_local_catalog"


def test_skills_endpoint_returns_no_fabricated_recommendations() -> None:
    """The shipped catalog is empty, so Jarvis recommends nothing."""
    skills = client.get("/skills").json()

    recommended = [skill for skill in skills if skill["status"] == "recommended"]
    assert recommended == []


def test_unknown_skill_decision_is_404() -> None:
    """You cannot approve a skill that does not exist."""
    response = client.post("/skills/not-a-skill/approve-research", json={})

    assert response.status_code == 404
