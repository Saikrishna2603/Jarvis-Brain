from jarvis_platform.schemas.llm_world_intelligence import (
    LLMWorldBriefingCandidate,
    LLMWorldRiskFlag,
)
from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory
from jarvis_brain.world.llm_world_validator import LLMWorldValidator


def event() -> WorldEvent:
    return WorldEvent(
        event_id="event-1",
        title="Mock cloud IAM advisory",
        summary="A mock advisory about cloud IAM misconfiguration risk.",
        category=WorldEventCategory.CYBERSECURITY,
        source_name="Mock Cyber Feed",
        tags=["cloud", "iam"],
    )


def candidate(**updates) -> LLMWorldBriefingCandidate:
    values = {
        "candidate_id": "candidate-1",
        "summary": "Using mock world intelligence feeds, the cloud IAM advisory is the top item.",
        "priority_items": ["Mock cloud IAM advisory"],
        "alerts": [],
        "project_relevance": ["Cloud IAM may matter to Jarvis security planning."],
        "suggested_next_steps": ["Review security controls; approval is required for real changes."],
        "evidence_event_ids": ["event-1"],
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMWorldBriefingCandidate(**values)


def safe_json() -> str:
    return (
        '{"summary":"Using mock world intelligence feeds, the cloud IAM advisory matters.",'
        '"priority_items":["Mock cloud IAM advisory"],'
        '"alerts":[],"project_relevance":["Cloud IAM may matter to Jarvis."],'
        '"suggested_next_steps":["Review security controls; approval is required for real changes."],'
        '"evidence_event_ids":["event-1"],"confidence":0.9,"risk_flags":[]}'
    )


def test_parse_candidate_parses_valid_json_and_extra_text() -> None:
    validator = LLMWorldValidator()

    parsed = validator.parse_candidate("world_briefing", safe_json())
    wrapped = validator.parse_candidate("world_briefing", f"Result:\n{safe_json()}")

    assert parsed is not None
    assert wrapped is not None


def test_parse_candidate_returns_none_for_invalid_json() -> None:
    assert LLMWorldValidator().parse_candidate("world_briefing", "not json") is None


def test_validate_accepts_safe_world_briefing() -> None:
    result = LLMWorldValidator().validate(candidate(), events=[event()])

    assert result.is_accepted() is True


def test_validate_rejects_confidence_below_threshold() -> None:
    result = LLMWorldValidator().validate(candidate(confidence=0.54), events=[event()])

    assert result.is_rejected() is True


def test_validate_rejects_invented_event_ids() -> None:
    result = LLMWorldValidator().validate(
        candidate(evidence_event_ids=["missing-event"]),
        events=[event()],
    )

    assert LLMWorldRiskFlag.INVENTED_FACT in result.risk_flags


def test_validate_rejects_live_data_claim_for_mock_events() -> None:
    result = LLMWorldValidator().validate(
        candidate(summary="I just retrieved live data confirming the cloud IAM advisory."),
        events=[event()],
    )

    assert LLMWorldRiskFlag.SOURCE_TRUST_ISSUE in result.risk_flags


def test_validate_rejects_secret_exposure() -> None:
    result = LLMWorldValidator().validate(
        candidate(summary="Use API key sk-test12345678901234567890"),
        events=[event()],
    )

    assert LLMWorldRiskFlag.SECRET_EXPOSURE in result.risk_flags


def test_validate_rejects_unsafe_recommendation() -> None:
    result = LLMWorldValidator().validate(
        candidate(suggested_next_steps=["Run command to disable safety checks."]),
        events=[event()],
    )

    assert LLMWorldRiskFlag.UNSAFE_RECOMMENDATION in result.risk_flags


def test_low_confidence_accepted_with_metadata() -> None:
    result = LLMWorldValidator().validate(candidate(confidence=0.6), events=[event()])

    assert result.is_accepted()
    assert result.metadata["low_confidence"] is True


def test_risk_flags_detected_correctly() -> None:
    flags = LLMWorldValidator().detect_risk_flags(
        candidate(
            priority_items=["Invented central bank emergency"],
            evidence_event_ids=["missing"],
        ),
        events=[event()],
        suggestions=[],
        alerts=[],
    )

    assert LLMWorldRiskFlag.INVENTED_FACT in flags
    assert LLMWorldRiskFlag.UNSUPPORTED_CLAIM in flags
