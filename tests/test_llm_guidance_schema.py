import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.llm_guidance import (
    LLMGuidanceCandidate,
    LLMGuidanceDecision,
    LLMGuidanceRiskFlag,
    LLMGuidanceValidationResult,
)


def candidate(**updates) -> LLMGuidanceCandidate:
    values = {
        "candidate_id": "candidate-1",
        "user_request": "Help with this issue",
        "domain": "coding",
        "summary": "Gather details before troubleshooting.",
        "response_text": "Please share the traceback and relevant code.",
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMGuidanceCandidate(**values)


def test_can_create_llm_guidance_candidate() -> None:
    assert candidate().domain == "coding"


@pytest.mark.parametrize("field", ["user_request", "summary", "response_text"])
def test_empty_required_text_validation_works(field: str) -> None:
    with pytest.raises(ValidationError):
        candidate(**{field: " "})


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_validation_works(confidence: float) -> None:
    with pytest.raises(ValidationError):
        candidate(confidence=confidence)


def test_validation_result_helpers_and_serialization() -> None:
    accepted = LLMGuidanceValidationResult(
        decision=LLMGuidanceDecision.ACCEPTED,
        candidate=candidate(),
        reason="Safe.",
    )
    rejected = LLMGuidanceValidationResult(
        decision=LLMGuidanceDecision.REJECTED,
        reason="Unsafe.",
        risk_flags=[LLMGuidanceRiskFlag.UNSAFE_INSTRUCTION],
    )

    assert accepted.is_accepted() is True
    assert rejected.is_rejected() is True
    assert rejected.has_risk_flags() is True
    assert accepted.model_dump(mode="json")["decision"] == "accepted"
