from app.knowledge.guidance_engine import GuidanceEngine
from app.knowledge.knowledge_gap_detector import KnowledgeGapDetector
from app.knowledge.llm_guidance_prompt_builder import LLMGuidancePromptBuilder
from jarvis_platform.schemas.evidence import EvidenceItem
from jarvis_platform.schemas.llm import LLMRole


def test_prompt_contains_guidance_safety_contract_and_context() -> None:
    request = "how do I fix my car thermostat"
    gaps = KnowledgeGapDetector().detect(request)
    base = GuidanceEngine().create_guidance(
        user_request=request,
        domain="automotive_repair",
        gaps=gaps,
    )
    evidence = EvidenceItem(
        evidence_id="evidence-1",
        title="Mock manual",
        summary="Mock thermostat safety summary.",
        metadata={"mock": True},
    )

    messages = LLMGuidancePromptBuilder().build_messages(
        user_request=request,
        domain="automotive_repair",
        base_guidance=base,
        gaps=gaps,
        evidence=[evidence],
        context={"location": "garage"},
    )

    assert [message.role for message in messages] == [LLMRole.SYSTEM, LLMRole.USER]
    system = messages[0].content
    user = messages[1].content
    assert "refining guidance only" in system
    assert "do not execute actions or call tools" in system
    assert "Return JSON only" in system
    assert "Do not open the coolant system while the engine is hot" in system
    assert "do not diagnose" in system
    assert "do not provide definitive legal advice" in system
    assert "Mock thermostat safety summary" in user
    assert "missing_fields" in user
    assert base.safety_warnings[0] in user
