from fastapi import APIRouter, Query

from jarvis_brain.agents.agent_lifecycle_dependencies import agent_event_broadcaster
from jarvis_platform.schemas.agent_narration import AgentActivityNarration

router = APIRouter(prefix="/agents/narration", tags=["agent-narration"])


@router.get("/status")
def get_agent_narration_status() -> dict[str, object]:
    """Return safe narration capability status."""
    return {
        "enabled": True,
        "deterministic_templates": True,
        "llm_generated": False,
        "humor_default": False,
        "chain_of_thought_exposed": False,
    }


@router.get("/recent", response_model=list[AgentActivityNarration])
def get_recent_agent_narrations(
    limit: int = Query(default=50, ge=1, le=200),
) -> list[AgentActivityNarration]:
    """Return retained safe narration records."""
    return agent_event_broadcaster.get_recent_narrations(limit)
