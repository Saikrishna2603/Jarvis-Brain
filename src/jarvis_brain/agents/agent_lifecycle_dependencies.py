import os

from jarvis_brain.agents.agent_event_broadcaster import AgentEventBroadcaster
from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_brain.agents.agent_stream_service import AgentStreamService


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    try:
        return max(1, int(os.getenv(name, str(default))))
    except ValueError:
        return default


AGENT_STREAM_ENABLED = _env_bool("AGENT_STREAM_ENABLED", True)
AGENT_STREAM_HEARTBEAT_SECONDS = _env_int(
    "AGENT_STREAM_HEARTBEAT_SECONDS", 15
)
AGENT_STREAM_QUEUE_SIZE = _env_int("AGENT_STREAM_QUEUE_SIZE", 200)

agent_event_broadcaster = AgentEventBroadcaster(
    queue_size=AGENT_STREAM_QUEUE_SIZE,
    heartbeat_seconds=AGENT_STREAM_HEARTBEAT_SECONDS,
)
agent_lifecycle_manager = AgentLifecycleManager(
    event_broadcaster=agent_event_broadcaster,
    max_recent_events=AGENT_STREAM_QUEUE_SIZE,
)
agent_stream_service = AgentStreamService(
    broadcaster=agent_event_broadcaster,
    lifecycle_manager=agent_lifecycle_manager,
    heartbeat_seconds=AGENT_STREAM_HEARTBEAT_SECONDS,
)
