from datetime import datetime

import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.agent_stream import (
    AgentStreamEnvelope,
    AgentStreamEventType,
    AgentStreamStatus,
)


def test_create_stream_envelope_and_demo_serialization() -> None:
    envelope = AgentStreamEnvelope(
        stream_event_id="stream-1",
        event_type=AgentStreamEventType.AGENT_CREATED,
        demo=True,
        data={"agent": {"name": "Forge-01"}},
    )

    serialized = envelope.model_dump(mode="json")
    assert serialized["demo"] is True
    assert datetime.fromisoformat(serialized["timestamp"]).tzinfo is not None


def test_empty_stream_event_id_fails() -> None:
    with pytest.raises(ValidationError):
        AgentStreamEnvelope(
            stream_event_id=" ",
            event_type=AgentStreamEventType.HEARTBEAT,
        )


def test_non_json_serializable_data_fails() -> None:
    with pytest.raises(ValidationError):
        AgentStreamEnvelope(
            stream_event_id="stream-1",
            event_type=AgentStreamEventType.ERROR,
            data={"bad": {"set"}},
        )


def test_stream_status_serializes() -> None:
    status = AgentStreamStatus(enabled=True, heartbeat_seconds=15)

    assert status.model_dump(mode="json") == {
        "enabled": True,
        "transport": "sse",
        "connected_clients": 0,
        "heartbeat_seconds": 15,
        "replay_supported": False,
        "storage": "in_memory",
        "metadata": {},
    }


def test_naive_timestamp_fails() -> None:
    with pytest.raises(ValidationError):
        AgentStreamEnvelope(
            stream_event_id="stream-1",
            event_type=AgentStreamEventType.CONNECTED,
            timestamp=datetime.now(),
        )
