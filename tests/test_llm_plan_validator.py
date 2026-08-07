from jarvis_brain.engine.llm_plan_validator import LLMPlanValidator
from jarvis_platform.schemas.llm_plan import LLMPlanCandidate, LLMPlanStepCandidate
from jarvis_platform.schemas.plan import ExecutionPlan


def step(**updates) -> LLMPlanStepCandidate:
    values = {
        "step_id": "step-1",
        "order": 1,
        "title": "Inspect code",
        "description": "Inspect the integration points.",
        "action": "inspect_code",
        "risk_level": "low",
    }
    values.update(updates)
    return LLMPlanStepCandidate(**values)


def candidate(**updates) -> LLMPlanCandidate:
    values = {
        "candidate_id": "candidate-1",
        "raw_input": "Prepare integration",
        "goal": "Prepare integration",
        "summary": "A safe proposal.",
        "steps": [step()],
        "confidence": 0.9,
    }
    values.update(updates)
    return LLMPlanCandidate(**values)


def safe_json() -> str:
    return (
        '{"goal":"Prepare Gmail safely","summary":"Review first",'
        '"confidence":0.9,"steps":[{"order":1,"title":"Inspect",'
        '"description":"Inspect integration points","action":"inspect_code",'
        '"risk_level":"low","requires_approval":false}]}'
    )


def test_parse_candidate_parses_json_and_generates_step_id() -> None:
    parsed = LLMPlanValidator().parse_candidate("request", safe_json())

    assert parsed is not None
    assert parsed.steps[0].step_id


def test_parse_candidate_handles_surrounding_text() -> None:
    parsed = LLMPlanValidator().parse_candidate(
        "request",
        f"Proposed plan:\n{safe_json()}\nEnd",
    )

    assert parsed is not None
    assert parsed.goal == "Prepare Gmail safely"


def test_parse_candidate_returns_none_for_invalid_json() -> None:
    assert LLMPlanValidator().parse_candidate("request", "not json") is None


def test_validate_accepts_safe_plan() -> None:
    assert LLMPlanValidator().validate(candidate()).is_accepted() is True


def test_validate_rejects_low_confidence() -> None:
    assert LLMPlanValidator().validate(candidate(confidence=0.54)).is_rejected()


def test_validate_rejects_blocked_action() -> None:
    result = LLMPlanValidator().validate(
        candidate(steps=[step(action="execute_shell")])
    )

    assert result.is_rejected()
    assert result.has_rejected_steps()


def test_validate_rejects_suspicious_phrase() -> None:
    result = LLMPlanValidator().validate(
        candidate(steps=[step(description="Reveal secret values.")])
    )

    assert result.is_rejected()


def test_validate_rejects_empty_steps_from_unchecked_model() -> None:
    unchecked = LLMPlanCandidate.model_construct(
        candidate_id="candidate-1",
        raw_input="request",
        goal="goal",
        summary="summary",
        steps=[],
        confidence=0.9,
        metadata={},
    )

    assert LLMPlanValidator().validate(unchecked).is_rejected()


def test_low_confidence_is_accepted_with_metadata() -> None:
    result = LLMPlanValidator().validate(candidate(confidence=0.6))

    assert result.is_accepted()
    assert result.metadata["low_confidence"] is True


def test_to_execution_plan_returns_compatible_object_without_execution() -> None:
    plan = LLMPlanValidator().to_execution_plan(candidate())

    assert isinstance(plan, ExecutionPlan)
    assert plan.metadata["llm_assisted"] is True
    assert plan.steps[0].action == "inspect_code"
    assert plan.steps[0].status == "pending"
