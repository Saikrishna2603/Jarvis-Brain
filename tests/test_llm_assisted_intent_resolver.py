from uuid import uuid4

from jarvis_brain.engine.llm_assisted_intent_resolver import (
    LLMAssistedIntentResolver,
    create_llm_assisted_intent_resolver,
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
            model="fake-intent-model",
            task_type=LLMTaskType.INTENT_RESOLUTION,
            status=self.status,
            content=self.content,
        )


def response_json(intent_type: str, action: str, confidence: float = 0.9) -> str:
    return (
        f'{{"intent_type":"{intent_type}","action":"{action}",'
        f'"target":"test","confidence":{confidence},"entities":{{}}}}'
    )


def test_high_confidence_rule_result_bypasses_llm() -> None:
    service = FakeLLMService(response_json("coding_help", "debug_code"))
    resolver = LLMAssistedIntentResolver(safe_llm_service=service, enabled=True)

    result = resolver.resolve("open youtube")

    assert result.action == "open_website"
    assert service.calls == 0


def test_unknown_rule_result_calls_llm_when_enabled() -> None:
    service = FakeLLMService(
        response_json("world_intelligence", "get_world_briefing")
    )
    resolver = LLMAssistedIntentResolver(safe_llm_service=service, enabled=True)

    result = resolver.resolve("tell me what happened globally today")

    assert result.intent_type == "world_intelligence"
    assert service.calls == 1


def test_disabled_resolver_returns_rule_result() -> None:
    service = FakeLLMService(response_json("coding_help", "debug_code"))
    resolver = LLMAssistedIntentResolver(safe_llm_service=service, enabled=False)

    result = resolver.resolve("ambiguous coding request")

    assert result.intent_type == "unknown"
    assert service.calls == 0


def test_accepted_coding_candidate_becomes_intent_result() -> None:
    service = FakeLLMService(response_json("coding_help", "debug_code"))
    resolver = LLMAssistedIntentResolver(safe_llm_service=service, enabled=True)

    result = resolver.resolve("please inspect this odd failure")

    assert result.intent_type == "coding_help"
    assert result.metadata["llm_assisted"] is True
    assert result.metadata["llm_model"] == "fake-intent-model"
    assert result.metadata["tools_executed"] is False


def test_llm_error_falls_back_to_rule_result() -> None:
    service = FakeLLMService("error", status=LLMStatus.ERROR)
    resolver = LLMAssistedIntentResolver(safe_llm_service=service, enabled=True)

    assert resolver.resolve("ambiguous").intent_type == "unknown"


def test_invalid_or_suspicious_llm_result_falls_back_safely() -> None:
    invalid = LLMAssistedIntentResolver(
        safe_llm_service=FakeLLMService("not json"),
        enabled=True,
    ).resolve("ambiguous")
    suspicious = LLMAssistedIntentResolver(
        safe_llm_service=FakeLLMService(
            response_json("coding_help", "execute_shell")
        ),
        enabled=True,
    ).resolve("ambiguous")

    assert invalid.intent_type == "unknown"
    assert invalid.metadata["llm_rejected"] is True
    assert suspicious.intent_type == "unknown"
    assert suspicious.metadata["llm_assisted"] is False


def test_factory_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("LLM_INTENT_ENABLED", raising=False)

    assert create_llm_assisted_intent_resolver().enabled is False


def test_factory_reads_intent_thresholds(monkeypatch) -> None:
    monkeypatch.setenv("LLM_INTENT_ENABLED", "true")
    monkeypatch.setenv("LLM_INTENT_RULE_CONFIDENCE_THRESHOLD", "0.9")
    monkeypatch.setenv("LLM_INTENT_CONFIDENCE_THRESHOLD", "0.6")

    resolver = create_llm_assisted_intent_resolver()

    assert resolver.enabled is True
    assert resolver.rule_confidence_threshold == 0.9
    assert resolver.llm_confidence_threshold == 0.6
