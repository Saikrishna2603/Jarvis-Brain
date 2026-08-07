from datetime import datetime

import pytest

from jarvis_platform.schemas.agent_narration import (
    AgentActivityNarration,
    AgentNarrationKind,
    AgentNarrationTone,
)


def test_can_create_narration() -> None:
    narration = AgentActivityNarration(
        narration_id="narration-1",
        agent_id="agent-1",
        kind=AgentNarrationKind.STATUS,
        tone=AgentNarrationTone.FOCUSED,
        text="Thinking through the next step.",
        source_event_type="status_changed",
    )

    assert narration.agent_id == "agent-1"
    assert narration.kind == AgentNarrationKind.STATUS


def test_empty_text_fails() -> None:
    with pytest.raises(ValueError):
        AgentActivityNarration(
            narration_id="narration-1",
            agent_id="agent-1",
            kind=AgentNarrationKind.STATUS,
            tone=AgentNarrationTone.NEUTRAL,
            text=" ",
            source_event_type="status_changed",
        )


def test_empty_agent_id_fails() -> None:
    with pytest.raises(ValueError):
        AgentActivityNarration(
            narration_id="narration-1",
            agent_id=" ",
            kind=AgentNarrationKind.STATUS,
            tone=AgentNarrationTone.NEUTRAL,
            text="Online.",
            source_event_type="agent_created",
        )


def test_humor_validation_works() -> None:
    with pytest.raises(ValueError):
        AgentActivityNarration(
            narration_id="narration-1",
            agent_id="agent-1",
            kind=AgentNarrationKind.HUMOR,
            tone=AgentNarrationTone.WARNING,
            text="A joke should not be here.",
            source_event_type="agent_failed",
            humor_used=True,
        )


def test_demo_flag_serializes() -> None:
    narration = AgentActivityNarration(
        narration_id="narration-1",
        agent_id="agent-1",
        kind=AgentNarrationKind.STATUS,
        tone=AgentNarrationTone.NEUTRAL,
        text="Coming online...",
        source_event_type="agent_created",
        demo=True,
    )

    assert narration.model_dump(mode="json")["demo"] is True


def test_timestamp_is_timezone_aware() -> None:
    narration = AgentActivityNarration(
        narration_id="narration-1",
        agent_id="agent-1",
        kind=AgentNarrationKind.STATUS,
        tone=AgentNarrationTone.NEUTRAL,
        text="Coming online...",
        source_event_type="agent_created",
    )

    assert isinstance(narration.created_at, datetime)
    assert narration.created_at.tzinfo is not None
