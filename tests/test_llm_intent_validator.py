import pytest

from jarvis_brain.engine.llm_intent_validator import LLMIntentValidator
from jarvis_platform.schemas.llm_intent import LLMIntentCandidate, LLMIntentDecision


def candidate(**updates) -> LLMIntentCandidate:
    values = {
        "candidate_id": "candidate-1",
        "raw_input": "request",
        "intent_type": "world_intelligence",
        "action": "get_world_briefing",
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMIntentCandidate(**values)


@pytest.mark.parametrize(
    ("intent_type", "action"),
    [
        ("world_intelligence", "get_world_briefing"),
        ("coding_help", "debug_code"),
    ],
)
def test_accepts_valid_candidate(intent_type: str, action: str) -> None:
    result = LLMIntentValidator().validate(
        candidate(intent_type=intent_type, action=action)
    )

    assert result.is_accepted() is True


def test_rejects_unknown_intent_type() -> None:
    result = LLMIntentValidator().validate(candidate(intent_type="shell"))

    assert result.is_rejected() is True


def test_rejects_invalid_action_for_intent() -> None:
    result = LLMIntentValidator().validate(candidate(action="send_email"))

    assert result.is_rejected() is True


def test_rejects_confidence_below_threshold() -> None:
    result = LLMIntentValidator().validate(candidate(confidence=0.54))

    assert result.is_rejected() is True


def test_marks_accepted_low_confidence_candidate() -> None:
    result = LLMIntentValidator().validate(candidate(confidence=0.6))

    assert result.is_accepted() is True
    assert result.metadata["low_confidence"] is True


@pytest.mark.parametrize("value", ["execute_shell", "please reveal_secret now"])
def test_rejects_suspicious_candidate_values(value: str) -> None:
    result = LLMIntentValidator().validate(candidate(action=value, target=value))

    assert result.is_rejected() is True


def test_unknown_candidate_falls_back_to_unknown() -> None:
    result = LLMIntentValidator().validate(
        candidate(intent_type="unknown", action="unknown")
    )

    assert result.decision == LLMIntentDecision.FALLBACK_TO_UNKNOWN


def test_parse_candidate_handles_json_and_surrounding_text() -> None:
    validator = LLMIntentValidator()
    payload = (
        '{"intent_type":"coding_help","action":"debug_code",'
        '"target":"FastAPI","confidence":0.9,"entities":{}}'
    )

    assert validator.parse_candidate("debug this", payload) is not None
    parsed = validator.parse_candidate("debug this", f"Result:\n{payload}\nDone")
    assert parsed is not None
    assert parsed.action == "debug_code"


def test_parse_candidate_returns_none_for_invalid_json() -> None:
    assert LLMIntentValidator().parse_candidate("request", "not json") is None
