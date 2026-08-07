import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.agent_lifecycle import (
    AgentEvent,
    AgentEventType,
    AgentLifecycleSnapshot,
    AgentLifecycleStatus,
    AgentRecord,
    AgentRole,
)


def test_create_agent_record() -> None:
    record = AgentRecord(
        agent_id="agent-1",
        name="Forge-01",
        role=AgentRole.PLANNER,
        purpose="Plan the work",
        status=AgentLifecycleStatus.CREATED,
    )

    assert record.name == "Forge-01"
    assert record.progress_percent == 0


def test_empty_agent_name_and_purpose_fail() -> None:
    for name, purpose in (("", "Plan"), ("Forge-01", "  ")):
        with pytest.raises(ValidationError):
            AgentRecord(
                agent_id="agent-1",
                name=name,
                role=AgentRole.PLANNER,
                purpose=purpose,
                status=AgentLifecycleStatus.CREATED,
            )


def test_progress_validation_works() -> None:
    with pytest.raises(ValidationError):
        AgentRecord(
            agent_id="agent-1",
            name="Forge-01",
            role=AgentRole.PLANNER,
            purpose="Plan",
            status=AgentLifecycleStatus.CREATED,
            progress_percent=101,
        )


def test_create_agent_event() -> None:
    event = AgentEvent(
        event_id="event-1",
        agent_id="agent-1",
        event_type=AgentEventType.AGENT_CREATED,
        message="Agent created.",
    )

    assert event.event_type == AgentEventType.AGENT_CREATED


def test_empty_event_message_fails() -> None:
    with pytest.raises(ValidationError):
        AgentEvent(
            event_id="event-1",
            agent_id="agent-1",
            event_type=AgentEventType.AGENT_CREATED,
            message=" ",
        )


def test_status_enum_validation() -> None:
    with pytest.raises(ValidationError):
        AgentRecord(
            agent_id="agent-1",
            name="Forge-01",
            role=AgentRole.PLANNER,
            purpose="Plan the work",
            status="not_a_status",
        )


def test_snapshot_serialization() -> None:
    snapshot = AgentLifecycleSnapshot(
        active_agents=[
            AgentRecord(
                agent_id="agent-1",
                name="Forge-01",
                role=AgentRole.PLANNER,
                purpose="Plan the work",
                status=AgentLifecycleStatus.WORKING,
            )
        ],
    )

    serialized = snapshot.model_dump(mode="json")

    assert serialized["active_agents"][0]["status"] == "working"
    assert serialized["archived_agents"] == []
    assert serialized["generated_at"]
