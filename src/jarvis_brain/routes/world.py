from pydantic import BaseModel
from fastapi import APIRouter

from jarvis_brain.world.llm_assisted_world_intelligence_engine import (
    create_llm_assisted_world_intelligence_engine,
)
from jarvis_brain.world.proactive_event_loop import ProactiveEventLoop


router = APIRouter()
proactive_event_loop = ProactiveEventLoop()
llm_assisted_world_engine = create_llm_assisted_world_intelligence_engine()


class WorldRunOnceRequest(BaseModel):
    """Optional request body for running one world intelligence cycle."""

    context: dict | None = None


@router.post("/world/run-once")
def run_world_once(request: WorldRunOnceRequest | None = None) -> dict:
    """Run one mock world intelligence collection cycle."""
    context = request.context if request is not None else None
    return proactive_event_loop.run_once(context=context)


@router.get("/world/briefing")
def get_world_briefing(use_llm: bool = False) -> dict:
    """Return a simple world intelligence briefing."""
    briefing = proactive_event_loop.get_daily_briefing()
    if not use_llm:
        return briefing

    events = proactive_event_loop.get_stored_events()
    suggestions = proactive_event_loop.latest_suggestions
    alerts = proactive_event_loop.get_alert_suggestions()
    refined = llm_assisted_world_engine.create_briefing(
        briefing_type="world_briefing",
        events=events,
        suggestions=suggestions,
        alerts=alerts,
        base_summary=(
            "Using mock world intelligence feeds, "
            f"{briefing['events_count']} events are available."
        ),
    )
    return {
        **briefing,
        "llm_summary": refined["summary"],
        "priority_items": refined["priority_items"],
        "llm_alerts": refined["alerts"],
        "project_relevance": refined["project_relevance"],
        "suggested_next_steps": refined["suggested_next_steps"],
        "evidence_event_ids": refined["evidence_event_ids"],
        "metadata": refined["metadata"],
    }


@router.get("/world/events")
def get_world_events(
    category: str | None = None,
    severity: str | None = None,
    high_priority_only: bool = False,
) -> list[dict]:
    """Return stored world events with optional route-level filters."""
    events = proactive_event_loop.get_stored_events()

    if category is not None:
        events = [event for event in events if event.category.value == category]

    if severity is not None:
        events = [event for event in events if event.severity.value == severity]

    if high_priority_only:
        events = [event for event in events if event.is_high_priority()]

    return [event.model_dump(mode="json") for event in events]


@router.get("/world/suggestions")
def get_world_suggestions() -> list[dict]:
    """Return the latest world intelligence suggestions."""
    return [
        suggestion.model_dump(mode="json")
        for suggestion in proactive_event_loop.latest_suggestions
    ]


@router.get("/world/alerts")
def get_world_alerts() -> list[dict]:
    """Return world intelligence suggestions that should alert the user."""
    return [
        suggestion.model_dump(mode="json")
        for suggestion in proactive_event_loop.get_alert_suggestions()
    ]
