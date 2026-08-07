from fastapi import APIRouter, HTTPException

from jarvis_brain.agents.agent_lifecycle_dependencies import (
    AGENT_STREAM_ENABLED,
    agent_lifecycle_manager,
)
from jarvis_platform.schemas.agent_lifecycle import AgentLifecycleSnapshot


router = APIRouter()


@router.get("/agents/lifecycle/status")
def get_agent_lifecycle_status() -> dict[str, object]:
    """Return lifecycle service status for the dashboard."""
    return {
        "enabled": True,
        "storage": "in_memory",
        "mode": "in_memory",
        "live_streaming": AGENT_STREAM_ENABLED,
        "demo_supported": True,
    }


@router.get("/agents/lifecycle/snapshot", response_model=AgentLifecycleSnapshot)
def get_agent_lifecycle_snapshot() -> AgentLifecycleSnapshot:
    """Return current agent lifecycle activity."""
    return agent_lifecycle_manager.get_snapshot()


@router.get("/agents/lifecycle/{agent_id}")
def get_agent_lifecycle_detail(agent_id: str) -> dict[str, object]:
    """Return safe lifecycle state and events for one agent."""
    agent = agent_lifecycle_manager.get_agent(agent_id)
    if agent is None:
        raise HTTPException(status_code=404, detail="Agent lifecycle record not found.")
    return {
        "agent": agent,
        "events": agent_lifecycle_manager.get_agent_events(agent_id),
    }


@router.post("/agents/lifecycle/demo", response_model=AgentLifecycleSnapshot)
def create_agent_lifecycle_demo() -> AgentLifecycleSnapshot:
    """Create clearly marked demo lifecycle data."""
    return agent_lifecycle_manager.create_demo_snapshot()


@router.post("/agents/lifecycle/reset-demo", response_model=AgentLifecycleSnapshot)
def reset_agent_lifecycle_demo() -> AgentLifecycleSnapshot:
    """Clear demo lifecycle data and return an empty snapshot."""
    return agent_lifecycle_manager.reset_demo_data()
