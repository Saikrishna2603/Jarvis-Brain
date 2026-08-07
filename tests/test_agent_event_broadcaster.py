import asyncio

import pytest

from jarvis_brain.agents.agent_event_broadcaster import AgentEventBroadcaster
from jarvis_platform.schemas.agent_lifecycle import (
    AgentEvent,
    AgentEventType,
    AgentLifecycleStatus,
    AgentRecord,
    AgentRole,
)
from jarvis_platform.schemas.agent_stream import AgentStreamEventType


@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.mark.anyio
async def test_subscribe_publish_and_unsubscribe() -> None:
    broadcaster = AgentEventBroadcaster()
    subscriber_id, queue = broadcaster.subscribe()

    broadcaster.publish(broadcaster.create_envelope(AgentStreamEventType.HEARTBEAT))
    envelope = await asyncio.wait_for(queue.get(), timeout=1)

    assert envelope.event_type == AgentStreamEventType.HEARTBEAT
    assert broadcaster.connected_client_count() == 1
    broadcaster.unsubscribe(subscriber_id)
    assert broadcaster.connected_client_count() == 0


@pytest.mark.anyio
async def test_multiple_subscribers_receive_event() -> None:
    broadcaster = AgentEventBroadcaster()
    first_id, first = broadcaster.subscribe()
    second_id, second = broadcaster.subscribe()
    broadcaster.publish(broadcaster.create_envelope(AgentStreamEventType.CONNECTED))

    assert (await asyncio.wait_for(first.get(), timeout=1)).event_type == AgentStreamEventType.CONNECTED
    assert (await asyncio.wait_for(second.get(), timeout=1)).event_type == AgentStreamEventType.CONNECTED
    broadcaster.unsubscribe(first_id)
    broadcaster.unsubscribe(second_id)


@pytest.mark.anyio
async def test_queue_overflow_drops_oldest_without_blocking() -> None:
    broadcaster = AgentEventBroadcaster(queue_size=1)
    subscriber_id, queue = broadcaster.subscribe()
    broadcaster.publish(broadcaster.create_envelope(AgentStreamEventType.CONNECTED))
    broadcaster.publish(broadcaster.create_envelope(AgentStreamEventType.HEARTBEAT))
    await asyncio.sleep(0)

    assert queue.qsize() == 1
    assert (await queue.get()).event_type == AgentStreamEventType.HEARTBEAT
    broadcaster.unsubscribe(subscriber_id)


@pytest.mark.anyio
async def test_lifecycle_publication_redacts_secrets() -> None:
    broadcaster = AgentEventBroadcaster()
    subscriber_id, queue = broadcaster.subscribe()
    secret = "sk-test123456789012345678901234567890"
    agent = AgentRecord(
        agent_id="agent-1",
        name="Sentinel-01",
        role=AgentRole.SECURITY,
        purpose=f"Review {secret}",
        status=AgentLifecycleStatus.WORKING,
    )
    event = AgentEvent(
        event_id="event-1",
        agent_id=agent.agent_id,
        event_type=AgentEventType.STATUS_CHANGED,
        message=f"Inspect token={secret}",
    )

    broadcaster.publish_lifecycle_event(event, agent)
    serialized = (await asyncio.wait_for(queue.get(), timeout=1)).model_dump_json()

    assert secret not in serialized
    assert "REDACTED" in serialized
    broadcaster.unsubscribe(subscriber_id)


def test_publish_isolates_sanitizer_failure() -> None:
    class BrokenFilter:
        def sanitize_metadata(self, metadata):
            raise RuntimeError("broken")

    broadcaster = AgentEventBroadcaster(safe_logging_filter=BrokenFilter())
    broadcaster.publish(broadcaster.create_envelope(AgentStreamEventType.ERROR))


@pytest.mark.anyio
async def test_lifecycle_event_also_publishes_narration() -> None:
    broadcaster = AgentEventBroadcaster()
    subscriber_id, queue = broadcaster.subscribe()
    agent = AgentRecord(
        agent_id="agent-1",
        name="Forge-01",
        role=AgentRole.PLANNER,
        purpose="Plan safely.",
        status=AgentLifecycleStatus.CREATED,
    )
    event = AgentEvent(
        event_id="event-1",
        agent_id=agent.agent_id,
        event_type=AgentEventType.AGENT_CREATED,
        message="created",
    )

    broadcaster.publish_lifecycle_event(event, agent)

    lifecycle = await asyncio.wait_for(queue.get(), timeout=1)
    narration = await asyncio.wait_for(queue.get(), timeout=1)
    assert lifecycle.event_type == AgentStreamEventType.LIFECYCLE_EVENT
    assert narration.event_type == AgentStreamEventType.AGENT_NARRATION
    assert narration.data["narration"]["text"] == "Coming online..."
    broadcaster.unsubscribe(subscriber_id)


@pytest.mark.anyio
async def test_demo_narration_is_marked() -> None:
    broadcaster = AgentEventBroadcaster()
    subscriber_id, queue = broadcaster.subscribe()
    agent = AgentRecord(
        agent_id="agent-1",
        name="Forge-01",
        role=AgentRole.PLANNER,
        purpose="Demo plan.",
        status=AgentLifecycleStatus.CREATED,
        metadata={"demo": True},
    )
    event = AgentEvent(
        event_id="event-1",
        agent_id=agent.agent_id,
        event_type=AgentEventType.AGENT_CREATED,
        message="created",
        metadata={"demo": True},
    )

    broadcaster.publish_lifecycle_event(event, agent)

    await asyncio.wait_for(queue.get(), timeout=1)
    narration = await asyncio.wait_for(queue.get(), timeout=1)
    assert narration.demo is True
    assert narration.data["narration"]["demo"] is True
    broadcaster.unsubscribe(subscriber_id)


@pytest.mark.anyio
async def test_narrator_exception_does_not_break_publish() -> None:
    class BrokenNarrator:
        def narrate(self, *_args, **_kwargs):
            raise RuntimeError("no narration")

    broadcaster = AgentEventBroadcaster(narrator=BrokenNarrator())  # type: ignore[arg-type]
    subscriber_id, queue = broadcaster.subscribe()
    event = AgentEvent(
        event_id="event-1",
        agent_id="agent-1",
        event_type=AgentEventType.AGENT_CREATED,
        message="created",
    )

    broadcaster.publish_lifecycle_event(event)

    lifecycle = await asyncio.wait_for(queue.get(), timeout=1)
    assert lifecycle.event_type == AgentStreamEventType.LIFECYCLE_EVENT
    broadcaster.unsubscribe(subscriber_id)
