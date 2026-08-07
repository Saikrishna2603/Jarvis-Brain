from jarvis_brain.engine.llm_intent_prompt_builder import LLMIntentPromptBuilder
from jarvis_platform.schemas.llm import LLMRole


def test_build_messages_contains_classifier_contract() -> None:
    messages = LLMIntentPromptBuilder().build_messages("Help me understand this")

    assert [message.role for message in messages] == [LLMRole.SYSTEM, LLMRole.USER]
    assert "Return JSON only" in messages[0].content
    assert "world_intelligence" in messages[0].content
    assert "never execute actions or call tools" in messages[0].content
    assert messages[1].content == "Help me understand this"
