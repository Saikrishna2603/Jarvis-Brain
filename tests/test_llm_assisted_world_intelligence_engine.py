from uuid import uuid4

from jarvis_platform.schemas.llm import (
    LLMProviderName,
    LLMResponse,
    LLMStatus,
    LLMTaskType,
)
from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory
from jarvis_brain.world.llm_assisted_world_intelligence_engine import (
    LLMAssistedWorldIntelligenceEngine,
    create_llm_assisted_world_intelligence_engine,
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
            model="fake-world-model",
            task_type=LLMTaskType.SUMMARIZATION,
            status=self.status,
            content=self.content,
        )


def event() -> WorldEvent:
    return WorldEvent(
        event_id="event-1",
        title="Mock cloud IAM advisory",
        summary="A mock advisory about cloud IAM misconfiguration risk.",
        category=WorldEventCategory.CYBERSECURITY,
        source_name="Mock Cyber Feed",
        tags=["cloud", "iam"],
    )


def safe_json() -> str:
    return (
        '{"summary":"Using mock world intelligence feeds, the cloud IAM advisory is the top item.",'
        '"priority_items":["Mock cloud IAM advisory"],'
        '"alerts":[],"project_relevance":["Cloud IAM may matter to Jarvis security planning."],'
        '"suggested_next_steps":["Review security controls; approval is required for real changes."],'
        '"evidence_event_ids":["event-1"],"confidence":0.9,"risk_flags":[]}'
    )


def test_disabled_engine_returns_base_briefing() -> None:
    service = FakeLLMService(safe_json())
    engine = LLMAssistedWorldIntelligenceEngine(
        safe_llm_service=service,
        enabled=False,
    )

    briefing = engine.create_briefing(events=[event()], base_summary="Base summary.")

    assert service.calls == 0
    assert briefing["summary"] == "Base summary."
    assert briefing["metadata"]["llm_assisted"] is False


def test_enabled_engine_returns_refined_briefing_with_metadata() -> None:
    service = FakeLLMService(safe_json())
    engine = LLMAssistedWorldIntelligenceEngine(
        safe_llm_service=service,
        enabled=True,
    )

    briefing = engine.create_briefing(events=[event()], base_summary="Base summary.")

    assert service.calls == 1
    assert briefing["metadata"]["llm_assisted_world"] is True
    assert briefing["metadata"]["world_model"] == "fake-world-model"
    assert briefing["metadata"]["tools_executed"] is False
    assert briefing["metadata"]["live_data_fetched"] is False
    assert "mock world intelligence feeds" in briefing["summary"].lower()


def test_error_invalid_and_unsafe_llm_output_fall_back() -> None:
    engines = [
        LLMAssistedWorldIntelligenceEngine(
            safe_llm_service=FakeLLMService("error", LLMStatus.ERROR),
            enabled=True,
        ),
        LLMAssistedWorldIntelligenceEngine(
            safe_llm_service=FakeLLMService("not json"),
            enabled=True,
        ),
        LLMAssistedWorldIntelligenceEngine(
            safe_llm_service=FakeLLMService(
                safe_json().replace(
                    "Using mock world intelligence feeds",
                    "I just retrieved live data",
                )
            ),
            enabled=True,
        ),
    ]

    for engine in engines:
        briefing = engine.create_briefing(events=[event()], base_summary="Base summary.")
        assert briefing["summary"] == "Base summary."
        assert briefing["metadata"]["llm_world_rejected"] is True


def test_factory_is_disabled_and_reads_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_WORLD_ENABLED", "false")
    assert create_llm_assisted_world_intelligence_engine().enabled is False

    monkeypatch.setenv("LLM_WORLD_ENABLED", "true")
    monkeypatch.setenv("LLM_WORLD_CONFIDENCE_THRESHOLD", "0.65")
    configured = create_llm_assisted_world_intelligence_engine()
    assert configured.enabled is True
    assert configured.llm_confidence_threshold == 0.65
