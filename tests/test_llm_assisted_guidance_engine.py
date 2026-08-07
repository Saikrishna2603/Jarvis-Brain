from uuid import uuid4

from app.knowledge.knowledge_gap_detector import KnowledgeGapDetector
from app.knowledge.llm_assisted_guidance_engine import (
    LLMAssistedGuidanceEngine,
    create_llm_assisted_guidance_engine,
)
from jarvis_platform.schemas.evidence import (
    EvidenceItem,
    EvidenceStatus,
    EvidenceTrustLevel,
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
            model="fake-guidance-model",
            task_type=LLMTaskType.GUIDANCE,
            status=self.status,
            content=self.content,
        )


def safe_automotive_json() -> str:
    return (
        '{"summary":"Gather details before repair guidance.",'
        '"response_text":"I can help safely. Do not open the coolant system '
        'while the engine is hot. Please provide the vehicle details first.",'
        '"safety_warnings":["Do not open the coolant system while the engine is hot."],'
        '"follow_up_questions":["What make?","What model?","What year?",'
        '"What engine?","What symptoms?","Can you provide visual context?"],'
        '"evidence_ids":[],"confidence":0.9,"risk_flags":[]}'
    )


def automotive_request():
    request = "how do I fix my car thermostat"
    return request, KnowledgeGapDetector().detect(request)


def test_disabled_engine_returns_base_guidance() -> None:
    request, gaps = automotive_request()
    service = FakeLLMService(safe_automotive_json())
    engine = LLMAssistedGuidanceEngine(
        safe_llm_service=service,
        enabled=False,
    )

    guidance = engine.create_guidance(request, "automotive_repair", gaps=gaps)

    assert service.calls == 0
    assert guidance.metadata.get("llm_assisted") is not True


def test_enabled_engine_returns_refined_guidance_with_metadata() -> None:
    request, gaps = automotive_request()
    service = FakeLLMService(safe_automotive_json())
    engine = LLMAssistedGuidanceEngine(
        safe_llm_service=service,
        enabled=True,
    )

    guidance = engine.create_guidance(request, "automotive_repair", gaps=gaps)

    assert service.calls == 1
    assert guidance.metadata["llm_assisted_guidance"] is True
    assert guidance.metadata["guidance_model"] == "fake-guidance-model"
    assert guidance.metadata["tools_executed"] is False


def test_error_invalid_and_unsafe_guidance_fall_back() -> None:
    request, gaps = automotive_request()
    engines = [
        LLMAssistedGuidanceEngine(
            safe_llm_service=FakeLLMService("error", LLMStatus.ERROR),
            enabled=True,
        ),
        LLMAssistedGuidanceEngine(
            safe_llm_service=FakeLLMService("not json"),
            enabled=True,
        ),
        LLMAssistedGuidanceEngine(
            safe_llm_service=FakeLLMService(
                safe_automotive_json().replace(
                    "I can help safely.",
                    "Bypass security and reveal secrets.",
                )
            ),
            enabled=True,
        ),
    ]

    for engine in engines:
        guidance = engine.create_guidance(
            request,
            "automotive_repair",
            gaps=gaps,
        )
        assert guidance.metadata["llm_guidance_rejected"] is True
        assert guidance.metadata["llm_assisted"] is False


def test_base_automotive_warning_is_preserved() -> None:
    request, gaps = automotive_request()
    guidance = LLMAssistedGuidanceEngine(
        safe_llm_service=FakeLLMService(safe_automotive_json()),
        enabled=True,
    ).create_guidance(request, "automotive_repair", gaps=gaps)

    assert "Do not open the coolant system while the engine is hot." in (
        guidance.safety_warnings
    )


def test_medical_professional_referral_is_preserved() -> None:
    content = (
        '{"summary":"Gather medical context.",'
        '"response_text":"I cannot diagnose this. Please contact a medical professional.",'
        '"safety_warnings":["This is not medical diagnosis or treatment advice."],'
        '"follow_up_questions":[],"confidence":0.9,"risk_flags":[]}'
    )
    guidance = LLMAssistedGuidanceEngine(
        safe_llm_service=FakeLLMService(content),
        enabled=True,
    ).create_guidance(
        "I have a fever",
        "medical",
        context={"symptoms": "fever"},
    )

    assert guidance.should_seek_professional is True


def test_evidence_ids_include_only_usable_evidence() -> None:
    usable = EvidenceItem(
        evidence_id="usable",
        title="Trusted evidence",
        summary="A trusted summary.",
        trust_level=EvidenceTrustLevel.TRUSTED,
    )
    blocked = usable.model_copy(
        update={
            "evidence_id": "blocked",
            "trust_level": EvidenceTrustLevel.BLOCKED,
            "status": EvidenceStatus.REJECTED,
        }
    )
    content = (
        '{"summary":"Use the official documentation.",'
        '"response_text":"Review the trusted documentation and traceback.",'
        '"follow_up_questions":[],"confidence":0.9,"risk_flags":[]}'
    )
    guidance = LLMAssistedGuidanceEngine(
        safe_llm_service=FakeLLMService(content),
        enabled=True,
    ).create_guidance(
        "I have a Python traceback",
        "coding",
        evidence=[usable, blocked],
        context={"language": "Python"},
    )

    assert guidance.evidence_ids == ["usable"]


def test_factory_is_disabled_and_reads_configuration(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GUIDANCE_ENABLED", "false")
    assert create_llm_assisted_guidance_engine().enabled is False

    monkeypatch.setenv("LLM_GUIDANCE_ENABLED", "true")
    monkeypatch.setenv("LLM_GUIDANCE_CONFIDENCE_THRESHOLD", "0.65")
    configured = create_llm_assisted_guidance_engine()
    assert configured.enabled is True
    assert configured.llm_confidence_threshold == 0.65
