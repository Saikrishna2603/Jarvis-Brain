from jarvis_brain.agents.agent_activity_narrator import AgentActivityNarrator
from jarvis_platform.schemas.agent_lifecycle import (
    AgentEvent,
    AgentEventType,
    AgentLifecycleStatus,
    AgentRecord,
    AgentRole,
)
from jarvis_platform.schemas.agent_narration import AgentNarrationKind, AgentNarrationTone


def make_event(
    event_type: AgentEventType,
    *,
    status: AgentLifecycleStatus | None = None,
    message: str = "event",
    metadata: dict | None = None,
) -> AgentEvent:
    return AgentEvent(
        event_id="event-1",
        agent_id="agent-1",
        event_type=event_type,
        message=message,
        status=status,
        metadata=metadata or {},
    )


def make_agent() -> AgentRecord:
    return AgentRecord(
        agent_id="agent-1",
        name="Forge-01",
        role=AgentRole.PLANNER,
        purpose="Plan safely.",
        status=AgentLifecycleStatus.THINKING,
    )


def test_created_event_maps_correctly() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(AgentEventType.AGENT_CREATED),
        make_agent(),
    )

    assert narration.text == "Coming online..."


def test_thinking_status_maps_correctly() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.STATUS_CHANGED,
            status=AgentLifecycleStatus.THINKING,
        ),
        make_agent(),
    )

    assert "safest approach" in narration.text


def test_waiting_approval_maps_correctly() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.STATUS_CHANGED,
            status=AgentLifecycleStatus.WAITING_FOR_APPROVAL,
        ),
        make_agent(),
    )

    assert narration.kind == AgentNarrationKind.APPROVAL
    assert "approval is required" in narration.text


def test_explicit_validation_reason_creates_reason_caption() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.STATUS_CHANGED,
            status=AgentLifecycleStatus.REVIEWING,
            metadata={"validation_reason": "The source needs verification."},
        ),
        make_agent(),
    )

    assert narration.kind == AgentNarrationKind.REASON
    assert narration.text == "Why this? The source needs verification."


def test_missing_reason_does_not_invent_explanation() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.STATUS_CHANGED,
            status=AgentLifecycleStatus.REVIEWING,
        ),
        make_agent(),
    )

    assert narration.kind != AgentNarrationKind.REASON
    assert not narration.text.startswith("Why this?")


def test_tool_name_is_sanitized() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.TOOL_CALLED,
            metadata={"tool_name": "token=SuperSecret123"},
        ),
        make_agent(),
    )

    assert "SuperSecret123" not in narration.text


def test_humor_is_off_by_default() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.AGENT_COMPLETED,
            status=AgentLifecycleStatus.COMPLETED,
        ),
        make_agent(),
    )

    assert narration.humor_used is False


def test_humor_works_in_safe_low_risk_context() -> None:
    narration = AgentActivityNarrator(humor_enabled=True).narrate(
        make_event(
            AgentEventType.AGENT_COMPLETED,
            status=AgentLifecycleStatus.COMPLETED,
        ),
        make_agent(),
    )

    assert narration.humor_used is True
    assert "explosions" in narration.text


def test_humor_is_suppressed_in_serious_context() -> None:
    narration = AgentActivityNarrator(humor_enabled=True).narrate(
        make_event(
            AgentEventType.AGENT_COMPLETED,
            status=AgentLifecycleStatus.COMPLETED,
            message="medical emergency completed",
        ),
        make_agent(),
    )

    assert narration.humor_used is False


def test_secrets_are_redacted() -> None:
    narration = AgentActivityNarrator().narrate(
        make_event(
            AgentEventType.STATUS_CHANGED,
            status=AgentLifecycleStatus.REVIEWING,
            metadata={"validation_reason": "password=SuperSecret123"},
        ),
        make_agent(),
    )

    assert "SuperSecret123" not in narration.text
    assert "[REDACTED_PASSWORD]" in narration.text
