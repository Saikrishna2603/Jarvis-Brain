import asyncio

import pytest

from jarvis_brain.agents.agent_event_broadcaster import AgentEventBroadcaster
from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_brain.agents.agent_stream_service import AgentStreamService
from jarvis_platform.schemas.agent_stream import AgentStreamEventType


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_stream_starts_connected_then_snapshot() -> None:
    broadcaster = AgentEventBroadcaster()
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)
    stream = AgentStreamService(broadcaster, manager).stream_events()

    connected = await anext(stream)
    snapshot = await anext(stream)

    assert "event: connected" in connected
    assert "event: snapshot" in snapshot
    assert '"active_agents":[]' in snapshot
    await stream.aclose()
    assert broadcaster.connected_client_count() == 0


@pytest.mark.anyio
async def test_published_event_appears_and_sse_is_valid() -> None:
    broadcaster = AgentEventBroadcaster()
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)
    service = AgentStreamService(broadcaster, manager)
    stream = service.stream_events(include_initial_snapshot=False)
    await anext(stream)
    envelope = broadcaster.create_envelope(AgentStreamEventType.AGENT_UPDATED)
    broadcaster.publish(envelope)

    message = await asyncio.wait_for(anext(stream), timeout=1)

    assert message.startswith(f"id: {envelope.stream_event_id}\n")
    assert "event: agent_updated\n" in message
    assert "\ndata: {" in message
    assert message.endswith("\n\n")
    await stream.aclose()


@pytest.mark.anyio
async def test_stream_emits_heartbeat_and_unsubscribes() -> None:
    broadcaster = AgentEventBroadcaster()
    manager = AgentLifecycleManager()
    stream = AgentStreamService(
        broadcaster, manager, heartbeat_seconds=0.01
    ).stream_events(include_initial_snapshot=False)
    await anext(stream)

    heartbeat = await asyncio.wait_for(anext(stream), timeout=1)
    assert "event: heartbeat" in heartbeat
    await stream.aclose()
    assert broadcaster.connected_client_count() == 0


@pytest.mark.anyio
async def test_stream_never_exposes_secret_payload() -> None:
    broadcaster = AgentEventBroadcaster()
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)
    stream = AgentStreamService(broadcaster, manager).stream_events(
        include_initial_snapshot=False
    )
    await anext(stream)
    secret = "sk-test123456789012345678901234567890"
    manager.create_agent("security", f"Review {secret}")

    messages = [await asyncio.wait_for(anext(stream), timeout=1) for _ in range(4)]
    assert secret not in "".join(messages)
    await stream.aclose()
