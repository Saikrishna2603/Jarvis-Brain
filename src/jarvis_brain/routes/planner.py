from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from jarvis_brain.routes.brain import brain_engine
from jarvis_platform.schemas.plan import ExecutionPlan


router = APIRouter()


class PlannerPreviewRequest(BaseModel):
    """Request for a non-executing plan preview."""

    raw_input: str
    use_llm: bool = False
    context: dict[str, Any] = Field(default_factory=dict)


@router.post("/planner/preview", response_model=ExecutionPlan)
def preview_plan(request: PlannerPreviewRequest) -> ExecutionPlan:
    """Create a plan preview without saving tasks or executing steps."""
    if request.use_llm:
        return brain_engine.llm_assisted_planner.create_plan(
            request.raw_input,
            context=request.context,
        )
    return brain_engine.planner.create_plan(request.raw_input)
