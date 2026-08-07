import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.llm_plan import (
    LLMPlanCandidate,
    LLMPlanDecision,
    LLMPlanStepCandidate,
    LLMPlanValidationResult,
)


def step(**updates) -> LLMPlanStepCandidate:
    values = {
        "step_id": "step-1",
        "order": 1,
        "title": "Inspect the project",
        "description": "Review the current integration boundaries.",
        "action": "inspect_code",
    }
    values.update(updates)
    return LLMPlanStepCandidate(**values)


def candidate(**updates) -> LLMPlanCandidate:
    values = {
        "candidate_id": "candidate-1",
        "raw_input": "Prepare an integration",
        "goal": "Prepare the integration safely",
        "summary": "Inspect, test, and document the integration.",
        "steps": [step()],
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMPlanCandidate(**values)


def test_can_create_llm_plan_step_candidate() -> None:
    assert step().order == 1


def test_step_order_validation_works() -> None:
    with pytest.raises(ValidationError):
        step(order=0)


@pytest.mark.parametrize("field", ["title", "description"])
def test_empty_step_text_validation_works(field: str) -> None:
    with pytest.raises(ValidationError):
        step(**{field: " "})


def test_can_create_llm_plan_candidate() -> None:
    assert candidate().goal == "Prepare the integration safely"


def test_empty_steps_validation_works() -> None:
    with pytest.raises(ValidationError):
        candidate(steps=[])


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_validation_works(confidence: float) -> None:
    with pytest.raises(ValidationError):
        candidate(confidence=confidence)


def test_validation_result_helpers_and_serialization() -> None:
    accepted = LLMPlanValidationResult(
        decision=LLMPlanDecision.ACCEPTED,
        candidate=candidate(),
        reason="Safe.",
        accepted_steps=[step()],
    )
    rejected = LLMPlanValidationResult(
        decision=LLMPlanDecision.REJECTED,
        reason="Unsafe.",
        rejected_steps=[step()],
    )

    assert accepted.is_accepted() is True
    assert rejected.is_rejected() is True
    assert rejected.has_rejected_steps() is True
    assert accepted.model_dump(mode="json")["decision"] == "accepted"
