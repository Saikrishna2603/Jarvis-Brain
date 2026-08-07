import asyncio
from dataclasses import dataclass
from threading import RLock
from typing import Any
from uuid import uuid4

from jarvis_brain.agents.agent_activity_narrator import AgentActivityNarrator
from jarvis_platform.schemas.agent_lifecycle import AgentEvent, AgentLifecycleSnapshot, AgentRecord
from jarvis_platform.schemas.agent_narration import AgentActivityNarration
from jarvis_platform.schemas.agent_stream import AgentStreamEnvelope, AgentStreamEventType
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter
from jarvis_platform.observability.event_logger import observability_event_logger
from jarvis_platform.observability.schemas import EventCategory, EventSeverity


@dataclass
class _Subscriber:
    queue: asyncio.Queue[AgentStreamEnvelope]
    loop: asyncio.AbstractEventLoop


class AgentEventBroadcaster:
    """Best-effort in-memory fan-out for sanitized lifecycle observations."""

    def __init__(
        self,
        queue_size: int = 200,
        heartbeat_seconds: int = 15,
        safe_logging_filter: SafeLoggingFilter | None = None,
        narrator: AgentActivityNarrator | None = None,
        max_recent_narrations: int = 200,
    ) -> None:
        """Create a broadcaster with bounded per-client queues."""
        self.queue_size = max(1, queue_size)
        self.heartbeat_seconds = max(1, heartbeat_seconds)
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()
        self.narrator = narrator or AgentActivityNarrator(
            safe_logging_filter=self.safe_logging_filter
        )
        self.max_recent_narrations = max(1, max_recent_narrations)
        self.recent_narrations: list[AgentActivityNarration] = []
        self._subscribers: dict[str, _Subscriber] = {}
        self._lock = RLock()

    def subscribe(self) -> tuple[str, asyncio.Queue[AgentStreamEnvelope]]:
        """Register a subscriber on the current event loop."""
        subscriber_id = f"stream_subscriber_{uuid4().hex}"
        queue: asyncio.Queue[AgentStreamEnvelope] = asyncio.Queue(
            maxsize=self.queue_size
        )
        subscriber = _Subscriber(queue=queue, loop=asyncio.get_running_loop())
        with self._lock:
            self._subscribers[subscriber_id] = subscriber
        return subscriber_id, queue

    def unsubscribe(self, subscriber_id: str) -> None:
        """Remove a subscriber if it still exists."""
        with self._lock:
            self._subscribers.pop(subscriber_id, None)

    def publish(self, envelope: AgentStreamEnvelope) -> None:
        """Publish without blocking lifecycle execution.

        When a queue is full, the oldest queued event is discarded so the
        client receives the newest state. REST snapshots remain authoritative.
        """
        try:
            safe_envelope = self._sanitize_envelope(envelope)
            with self._lock:
                subscribers = list(self._subscribers.values())
            for subscriber in subscribers:
                try:
                    subscriber.loop.call_soon_threadsafe(
                        self._enqueue_latest,
                        subscriber.queue,
                        safe_envelope,
                    )
                except (RuntimeError, AttributeError):
                    continue
        except Exception:
            return

    def publish_lifecycle_event(
        self,
        agent_event: AgentEvent,
        agent_record: AgentRecord | None = None,
    ) -> None:
        """Convert one lifecycle record into a safe stream envelope."""
        data: dict[str, Any] = {
            "event": agent_event.model_dump(mode="json"),
        }
        if agent_record is not None:
            data["agent"] = agent_record.model_dump(mode="json")
        self.publish(
            AgentStreamEnvelope(
                stream_event_id=f"stream_{uuid4().hex}",
                event_type=AgentStreamEventType.LIFECYCLE_EVENT,
                agent_id=agent_event.agent_id,
                lifecycle_event_id=agent_event.event_id,
                data=data,
                demo=bool(
                    agent_event.metadata.get("demo")
                    or (agent_record and agent_record.metadata.get("demo"))
                ),
            )
        )
        failed = agent_event.event_type.value == "agent_failed"
        observability_event_logger.log_event(
            agent_event.event_type.value,
            agent_event.message,
            metadata={
                "agent_id": agent_event.agent_id,
                "status": agent_event.status.value if agent_event.status else None,
                "demo": bool(agent_event.metadata.get("demo")),
            },
            category=EventCategory.AGENTS,
            title=agent_event.event_type.value.replace("_", " ").title(),
            severity=EventSeverity.ERROR if failed else EventSeverity.INFO,
        )
        self.publish_narration_for_event(agent_event, agent_record)

    def publish_narration_for_event(
        self,
        agent_event: AgentEvent,
        agent_record: AgentRecord | None = None,
    ) -> None:
        """Publish safe UI narration after the source lifecycle event."""
        try:
            narration = self.narrator.narrate(agent_event, agent_record)
            with self._lock:
                self.recent_narrations.append(narration)
                self.recent_narrations = self.recent_narrations[
                    -self.max_recent_narrations :
                ]
            self.publish(
                AgentStreamEnvelope(
                    stream_event_id=f"stream_{uuid4().hex}",
                    event_type=AgentStreamEventType.AGENT_NARRATION,
                    agent_id=narration.agent_id,
                    lifecycle_event_id=narration.lifecycle_event_id,
                    data={"narration": narration.model_dump(mode="json")},
                    demo=narration.demo,
                )
            )
        except Exception:
            return

    def publish_snapshot(self, snapshot: AgentLifecycleSnapshot) -> None:
        """Publish a complete source-of-truth snapshot."""
        self.publish(
            AgentStreamEnvelope(
                stream_event_id=f"stream_{uuid4().hex}",
                event_type=AgentStreamEventType.SNAPSHOT,
                data=snapshot.model_dump(mode="json"),
                demo=bool(snapshot.metadata.get("demo")),
            )
        )

    def publish_heartbeat(self) -> None:
        """Publish an observational heartbeat."""
        self.publish(self.create_envelope(AgentStreamEventType.HEARTBEAT))

    def create_envelope(
        self,
        event_type: AgentStreamEventType,
        *,
        agent: AgentRecord | None = None,
        data: dict[str, Any] | None = None,
        demo: bool | None = None,
    ) -> AgentStreamEnvelope:
        """Build a typed envelope for an agent state transition."""
        payload = dict(data or {})
        if agent is not None:
            payload["agent"] = agent.model_dump(mode="json")
        return AgentStreamEnvelope(
            stream_event_id=f"stream_{uuid4().hex}",
            event_type=event_type,
            agent_id=agent.agent_id if agent else None,
            data=payload,
            demo=bool(agent.metadata.get("demo")) if demo is None and agent else bool(demo),
        )

    def connected_client_count(self) -> int:
        """Return the current number of registered clients."""
        with self._lock:
            return len(self._subscribers)

    def get_recent_narrations(self, limit: int = 50) -> list[AgentActivityNarration]:
        """Return retained safe narrations, newest first."""
        safe_limit = max(1, min(limit, self.max_recent_narrations))
        with self._lock:
            return list(reversed(self.recent_narrations[-safe_limit:]))

    def _sanitize_envelope(
        self, envelope: AgentStreamEnvelope
    ) -> AgentStreamEnvelope:
        return envelope.model_copy(
            update={
                "data": self.safe_logging_filter.sanitize_metadata(envelope.data),
                "metadata": self.safe_logging_filter.sanitize_metadata(
                    envelope.metadata
                ),
            }
        )

    @staticmethod
    def _enqueue_latest(
        queue: asyncio.Queue[AgentStreamEnvelope],
        envelope: AgentStreamEnvelope,
    ) -> None:
        try:
            if queue.full():
                queue.get_nowait()
            queue.put_nowait(envelope)
        except (asyncio.QueueEmpty, asyncio.QueueFull):
            return
