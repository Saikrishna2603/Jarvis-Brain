from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from jarvis_brain.agents.swarm_coordinator import SwarmCoordinator
from jarvis_brain.routes.brain import brain_engine
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter


router = APIRouter()
swarm_coordinator = SwarmCoordinator()
safe_logging_filter = SafeLoggingFilter()


class SwarmRequest(BaseModel):
    """Request body for swarm preview and safe run."""

    raw_input: str
    context: dict[str, Any] = Field(default_factory=dict)


@router.post("/swarm/preview")
def swarm_preview(request: SwarmRequest) -> dict:
    """Return a proposal-only multi-agent analysis."""
    safe_context = safe_logging_filter.sanitize_metadata(request.context)
    result = swarm_coordinator.preview(
        raw_input=safe_logging_filter.sanitize_message(request.raw_input),
        context=safe_context,
    )
    _audit_swarm("swarm_preview", result)
    return result


@router.post("/swarm/run-safe")
def swarm_run_safe(request: SwarmRequest) -> dict:
    """Return reviewed proposals through the safe swarm path."""
    safe_context = safe_logging_filter.sanitize_metadata(request.context)
    result = swarm_coordinator.run_safe(
        raw_input=safe_logging_filter.sanitize_message(request.raw_input),
        context=safe_context,
    )
    _audit_swarm("swarm_run_safe", result)
    return result


@router.get("/swarm/status")
def swarm_status() -> dict:
    """Return swarm runtime status."""
    return swarm_coordinator.status()


def _audit_swarm(event_type: str, result: dict) -> None:
    """Record a sanitized swarm audit event."""
    brain_engine.audit_manager.record_event(
        event_type=event_type,
        message=f"Swarm event: {event_type}",
        metadata=safe_logging_filter.sanitize_metadata(
            {
                "mode": result["mode"],
                "executed": result["executed"],
                "accepted_count": result["accepted_count"],
                "rejected_count": result["rejected_count"],
            }
        ),
    )
