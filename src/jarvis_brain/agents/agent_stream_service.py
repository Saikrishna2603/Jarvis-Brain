import asyncio
import json
from collections.abc import AsyncIterator
from uuid import uuid4

from jarvis_brain.agents.agent_event_broadcaster import AgentEventBroadcaster
from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_platform.schemas.agent_stream import AgentStreamEnvelope, AgentStreamEventType


class AgentStreamService:
    """Generate an observational SSE stream from lifecycle broadcasts."""

    def __init__(
        self,
        broadcaster: AgentEventBroadcaster,
        lifecycle_manager: AgentLifecycleManager,
        heartbeat_seconds: float = 15,
    ) -> None:
        self.broadcaster = broadcaster
        self.lifecycle_manager = lifecycle_manager
        self.heartbeat_seconds = max(0.01, heartbeat_seconds)

    async def stream_events(
        self, include_initial_snapshot: bool = True
    ) -> AsyncIterator[str]:
        """Yield connected, snapshot, lifecycle, and heartbeat SSE messages."""
        subscriber_id, queue = self.broadcaster.subscribe()
        try:
            connected = AgentStreamEnvelope(
                stream_event_id=f"stream_{uuid4().hex}",
                event_type=AgentStreamEventType.CONNECTED,
                data={"transport": "sse", "replay_supported": False},
            )
            yield self.format_sse(connected)
            if include_initial_snapshot:
                current_snapshot = self.lifecycle_manager.get_snapshot()
                snapshot = AgentStreamEnvelope(
                    stream_event_id=f"stream_{uuid4().hex}",
                    event_type=AgentStreamEventType.SNAPSHOT,
                    data=current_snapshot.model_dump(mode="json"),
                    demo=bool(current_snapshot.metadata.get("demo")),
                )
                yield self.format_sse(snapshot)

            while True:
                try:
                    envelope = await asyncio.wait_for(
                        queue.get(), timeout=self.heartbeat_seconds
                    )
                except TimeoutError:
                    envelope = AgentStreamEnvelope(
                        stream_event_id=f"stream_{uuid4().hex}",
                        event_type=AgentStreamEventType.HEARTBEAT,
                    )
                yield self.format_sse(envelope)
        except asyncio.CancelledError:
            raise
        finally:
            self.broadcaster.unsubscribe(subscriber_id)

    @staticmethod
    def format_sse(envelope: AgentStreamEnvelope) -> str:
        """Encode one envelope using valid Server-Sent Events framing."""
        payload = json.dumps(envelope.model_dump(mode="json"), separators=(",", ":"))
        return (
            f"id: {envelope.stream_event_id}\n"
            f"event: {envelope.event_type.value}\n"
            f"data: {payload}\n\n"
        )
