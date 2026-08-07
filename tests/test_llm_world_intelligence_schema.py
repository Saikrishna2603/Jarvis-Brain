import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.llm_world_intelligence import (
    LLMWorldBriefingCandidate,
    LLMWorldDecision,
    LLMWorldRiskFlag,
    LLMWorldValidationResult,
)


def test_can_create_llm_world_briefing_candidate() -> None:
    candidate = LLMWorldBriefingCandidate(
        candidate_id="candidate-1",
        summary="Using mock feeds, one cyber item matters most.",
        confidence=0.9,
    )

    assert candidate.briefing_type == "world_briefing"
    assert candidate.priority_items == []


def test_empty_summary_validation_works() -> None:
    with pytest.raises(ValidationError):
        LLMWorldBriefingCandidate(
            candidate_id="candidate-1",
            summary=" ",
            confidence=0.9,
        )


@pytest.mark.parametrize("confidence", [-0.1, 1.1])
def test_confidence_validation_works(confidence: float) -> None:
    with pytest.raises(ValidationError):
        LLMWorldBriefingCandidate(
            candidate_id="candidate-1",
            summary="Summary.",
            confidence=confidence,
        )


def test_validation_result_helpers_and_serialization() -> None:
    accepted = LLMWorldValidationResult(
        decision=LLMWorldDecision.ACCEPTED,
        reason="ok",
    )
    rejected = LLMWorldValidationResult(
        decision=LLMWorldDecision.REJECTED,
        reason="bad",
        risk_flags=[LLMWorldRiskFlag.UNSUPPORTED_CLAIM],
    )

    assert accepted.is_accepted() is True
    assert rejected.is_rejected() is True
    assert rejected.has_risk_flags() is True
    assert rejected.model_dump(mode="json")["risk_flags"] == ["unsupported_claim"]
