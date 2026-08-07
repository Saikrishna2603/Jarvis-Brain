from app.knowledge.knowledge_gap_detector import KnowledgeGapDetector
from app.knowledge.llm_guidance_validator import LLMGuidanceValidator
from jarvis_platform.schemas.llm_guidance import LLMGuidanceCandidate, LLMGuidanceRiskFlag


def candidate(**updates) -> LLMGuidanceCandidate:
    values = {
        "candidate_id": "candidate-1",
        "user_request": "how do I fix my car thermostat",
        "domain": "automotive_repair",
        "summary": "Gather vehicle details and inspect safely.",
        "response_text": (
            "Do not open the coolant system while the engine is hot. "
            "I need vehicle details before exact guidance."
        ),
        "follow_up_questions": [
            "What make do you have?",
            "What model do you have?",
            "What year is it?",
            "What engine is installed?",
            "What symptoms are present?",
            "Can you provide visual context?",
        ],
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMGuidanceCandidate(**values)


def safe_json() -> str:
    return (
        '{"summary":"Gather details safely","response_text":"Do not open the '
        'coolant system while the engine is hot.","follow_up_questions":'
        '["What make?","What model?","What year?","What engine?",'
        '"What symptoms?","Can you provide visual context?"],'
        '"confidence":0.9,"risk_flags":[]}'
    )


def automotive_gaps():
    return KnowledgeGapDetector().detect("how do I fix my car thermostat")


def test_parse_candidate_parses_json_and_surrounding_text() -> None:
    validator = LLMGuidanceValidator()

    parsed = validator.parse_candidate(
        "how do I fix my car thermostat",
        "automotive_repair",
        safe_json(),
    )
    wrapped = validator.parse_candidate(
        "how do I fix my car thermostat",
        "automotive_repair",
        f"Result:\n{safe_json()}\nEnd",
    )

    assert parsed is not None
    assert wrapped is not None


def test_parse_candidate_returns_none_for_invalid_json() -> None:
    assert (
        LLMGuidanceValidator().parse_candidate("request", "coding", "not json")
        is None
    )


def test_validate_accepts_safe_automotive_clarification() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(),
        domain="automotive_repair",
        gaps=automotive_gaps(),
    )

    assert result.is_accepted()


def test_rejects_exact_automotive_steps_without_vehicle_context() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(response_text="Remove the thermostat and torque it to specification."),
        domain="automotive_repair",
        gaps=automotive_gaps(),
    )

    assert result.is_rejected()


def test_rejects_coolant_guidance_without_hot_engine_warning() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(response_text="Check the coolant level."),
        domain="automotive_repair",
        gaps=automotive_gaps(),
    )

    assert result.is_rejected()


def test_rejects_medical_diagnosis() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(
            user_request="I have a fever",
            domain="medical",
            response_text="Your diagnosis is an infection.",
            follow_up_questions=[],
        ),
        domain="medical",
    )

    assert LLMGuidanceRiskFlag.PROFESSIONAL_ADVICE_RISK in result.risk_flags


def test_rejects_legal_definitive_advice() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(
            domain="legal",
            response_text="This is definitive legal advice.",
            follow_up_questions=[],
        ),
        domain="legal",
    )

    assert result.is_rejected()


def test_rejects_finance_guarantee() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(
            domain="finance",
            response_text="This investment has a guaranteed return.",
            follow_up_questions=[],
        ),
        domain="finance",
    )

    assert result.is_rejected()


def test_rejects_cybersecurity_offensive_steps() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(
            domain="cybersecurity",
            response_text="Exploit the target and deploy payload.",
            follow_up_questions=[],
        ),
        domain="cybersecurity",
    )

    assert result.is_rejected()


def test_rejects_secret_exposure() -> None:
    result = LLMGuidanceValidator().validate(
        candidate(response_text="Use key sk-test12345678901234567890"),
        domain="automotive_repair",
    )

    assert LLMGuidanceRiskFlag.SECRET_EXPOSURE in result.risk_flags


def test_low_confidence_accepted_and_below_threshold_rejected() -> None:
    validator = LLMGuidanceValidator()
    low = validator.validate(candidate(confidence=0.6), "automotive_repair")
    rejected = validator.validate(candidate(confidence=0.54), "automotive_repair")

    assert low.is_accepted()
    assert low.metadata["low_confidence"] is True
    assert rejected.is_rejected()
