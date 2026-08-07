from uuid import uuid4

from jarvis_brain.engine.llm_assisted_planner import (
    LLMAssistedPlanner,
    create_llm_assisted_planner,
)
from jarvis_platform.schemas.llm import (
    LLMProviderName,
    LLMResponse,
    LLMStatus,
    LLMTaskType,
)


class FakeLLMService:
    def __init__(self, content: str, status: LLMStatus = LLMStatus.SUCCESS) -> None:
        self.content = content
        self.status = status
        self.calls = 0

    def generate(self, messages, metadata=None) -> LLMResponse:
        self.calls += 1
        return LLMResponse(
            response_id=str(uuid4()),
            request_id=str(uuid4()),
            provider=LLMProviderName.MOCK,
            model="fake-planning-model",
            task_type=LLMTaskType.PLANNING,
            status=self.status,
            content=self.content,
        )


def safe_plan_json() -> str:
    return (
        '{"goal":"Prepare Jarvis for Gmail integration safely",'
        '"summary":"Inspect boundaries, propose changes, and test safely.",'
        '"confidence":0.9,"steps":['
        '{"order":1,"title":"Analyze requirements",'
        '"description":"Identify Gmail integration boundaries.",'
        '"action":"analyze_request","risk_level":"low",'
        '"requires_approval":false},'
        '{"order":2,"title":"Review security",'
        '"description":"Review OAuth and SecretGuard requirements.",'
        '"action":"review_security","risk_level":"medium",'
        '"requires_approval":false},'
        '{"order":3,"title":"Propose changes",'
        '"description":"Propose code changes without applying them.",'
        '"action":"propose_code_change","risk_level":"medium",'
        '"requires_approval":false},'
        '{"order":4,"title":"Plan tests",'
        '"description":"Define deterministic integration tests.",'
        '"action":"run_tests","risk_level":"low",'
        '"requires_approval":false}]}'
    )


def test_disabled_assisted_planner_returns_rule_plan() -> None:
    service = FakeLLMService(safe_plan_json())
    planner = LLMAssistedPlanner(safe_llm_service=service, enabled=False)

    plan = planner.create_plan("review my finances")

    assert [step.action for step in plan.steps] == [
        "list_accounts",
        "summarize_spending",
        "detect_subscriptions",
    ]
    assert service.calls == 0


def test_enabled_planner_returns_validated_llm_plan() -> None:
    service = FakeLLMService(safe_plan_json())
    planner = LLMAssistedPlanner(safe_llm_service=service, enabled=True)

    plan = planner.create_plan("prepare Jarvis for Gmail integration safely")

    assert service.calls == 1
    assert plan.metadata["llm_assisted"] is True
    assert plan.metadata["llm_model"] == "fake-planning-model"
    assert len(plan.steps) == 4
    assert all(step.status == "pending" for step in plan.steps)


def test_llm_error_falls_back_to_rule_plan() -> None:
    planner = LLMAssistedPlanner(
        safe_llm_service=FakeLLMService("error", status=LLMStatus.ERROR),
        enabled=True,
    )

    plan = planner.create_plan("review my finances")

    assert plan.steps
    assert plan.metadata["llm_plan_rejected"] is True


def test_invalid_json_falls_back_to_rule_plan() -> None:
    planner = LLMAssistedPlanner(
        safe_llm_service=FakeLLMService("not json"),
        enabled=True,
    )

    plan = planner.create_plan("review my finances")

    assert plan.metadata["llm_assisted"] is False


def test_blocked_plan_falls_back_without_execution() -> None:
    blocked = safe_plan_json().replace("analyze_request", "execute_shell", 1)
    planner = LLMAssistedPlanner(
        safe_llm_service=FakeLLMService(blocked),
        enabled=True,
    )

    plan = planner.create_plan("review my finances")

    assert plan.metadata["llm_plan_rejected"] is True
    assert all(step.action != "execute_shell" for step in plan.steps)


def test_factory_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PLANNER_ENABLED", raising=False)

    assert create_llm_assisted_planner().enabled is False


def test_factory_reads_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PLANNER_ENABLED", "true")
    monkeypatch.setenv("LLM_PLANNER_CONFIDENCE_THRESHOLD", "0.65")

    planner = create_llm_assisted_planner()

    assert planner.enabled is True
    assert planner.llm_confidence_threshold == 0.65
