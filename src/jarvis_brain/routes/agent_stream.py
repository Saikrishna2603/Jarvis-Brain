from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from jarvis_brain.agents.agent_lifecycle_dependencies import (
    AGENT_STREAM_ENABLED,
    AGENT_STREAM_HEARTBEAT_SECONDS,
    agent_event_broadcaster,
    agent_stream_service,
)
from jarvis_brain.agents.agent_stream_service import AgentStreamService
from jarvis_platform.schemas.agent_stream import AgentStreamStatus


router = APIRouter()


def get_agent_stream_service() -> AgentStreamService:
    """Return the shared observational stream service."""
    return agent_stream_service


@router.get("/agents/stream/status", response_model=AgentStreamStatus)
def get_agent_stream_status() -> AgentStreamStatus:
    """Return safe stream diagnostics without subscriber identifiers."""
    return AgentStreamStatus(
        enabled=AGENT_STREAM_ENABLED,
        connected_clients=agent_event_broadcaster.connected_client_count(),
        heartbeat_seconds=AGENT_STREAM_HEARTBEAT_SECONDS,
        metadata={"source_of_truth": "rest_snapshot", "observational_only": True},
    )


@router.get("/agents/stream/events")
async def get_agent_stream_events(
    include_snapshot: bool = True,
    service: AgentStreamService = Depends(get_agent_stream_service),
) -> StreamingResponse:
    """Open a read-only Server-Sent Events lifecycle stream."""
    if not AGENT_STREAM_ENABLED:
        raise HTTPException(status_code=503, detail="Agent event streaming is disabled.")
    return StreamingResponse(
        service.stream_events(include_initial_snapshot=include_snapshot),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
