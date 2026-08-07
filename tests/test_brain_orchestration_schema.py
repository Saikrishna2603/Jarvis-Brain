import pytest

from jarvis_platform.schemas.brain_orchestration import (
    BrainCapability,
    BrainEventType,
    BrainExecutionNodeType,
    BrainIntentCandidate,
    BrainIntentType,
    BrainOrchestrationEvent,
    BrainOrchestratorProposal,
    BrainPlanStep,
    BrainToolProposal,
)


def test_create_brain_orchestrator_proposal() -> None:
    proposal = BrainOrchestratorProposal(
        summary="Plan a safe response.",
        intents=[
            BrainIntentCandidate(
                intent_type=BrainIntentType.CODING,
                confidence=0.8,
            )
        ],
        plan_steps=[
            BrainPlanStep(
                step_id="step-1",
                order=1,
                title="Inspect issue",
                description="Ask for traceback and code context.",
                node_type=BrainExecutionNodeType.PLANNING,
            )
        ],
        confidence=0.8,
    )

    assert proposal.intents[0].intent_type == BrainIntentType.CODING


def test_empty_proposal_summary_fails() -> None:
    with pytest.raises(ValueError):
        BrainOrchestratorProposal(summary="", confidence=0.8)


def test_tool_proposal_action_validation() -> None:
    with pytest.raises(ValueError):
        BrainToolProposal(action="")


def test_plan_step_validation() -> None:
    with pytest.raises(ValueError):
        BrainPlanStep(step_id="step-1", order=1, title="", description="desc")


def test_brain_event_serializes() -> None:
    event = BrainOrchestrationEvent(
        event_id="event-1",
        request_id="request-1",
        event_type=BrainEventType.BRAIN_STARTED,
        message="Started.",
        metadata={"capability": BrainCapability.REASONING.value},
    )

    data = event.model_dump(mode="json")

    assert data["event_type"] == "BrainStarted"
    assert data["created_at"].endswith("Z") or data["created_at"].endswith("+00:00")
