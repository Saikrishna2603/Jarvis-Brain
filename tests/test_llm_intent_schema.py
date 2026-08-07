import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.llm_intent import (
    LLMIntentCandidate,
    LLMIntentDecision,
    LLMIntentValidationResult,
)


def make_candidate(**updates) -> LLMIntentCandidate:
    values = {
        "candidate_id": "candidate-1",
        "raw_input": "What happened today?",
        "intent_type": "world_intelligence",
        "action": "get_world_briefing",
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMIntentCandidate(**values)


def test_can_create_llm_intent_candidate() -> None:
    assert make_candidate().intent_type == "world_intelligence"


@pytest.mark.parametrize("field", ["raw_input", "intent_type"])
def test_candidate_rejects_empty_required_text(field: str) -> None:
    with pytest.raises(ValidationError):
        make_candidate(**{field: " "})


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_candidate_validates_confidence(confidence: float) -> None:
    with pytest.raises(ValidationError):
        make_candidate(confidence=confidence)


def test_validation_result_helpers_and_serialization() -> None:
    accepted = LLMIntentValidationResult(
        decision=LLMIntentDecision.ACCEPTED,
        candidate=make_candidate(),
        reason="Allowed.",
    )
    rejected = LLMIntentValidationResult(
        decision=LLMIntentDecision.REJECTED,
        reason="Rejected.",
    )

    assert accepted.is_accepted() is True
    assert accepted.is_rejected() is False
    assert rejected.is_rejected() is True
    assert accepted.model_dump(mode="json")["decision"] == "accepted"
