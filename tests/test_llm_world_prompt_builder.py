from jarvis_platform.schemas.llm import LLMRole
from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory
from jarvis_platform.schemas.world_suggestion import WorldSuggestion
from jarvis_brain.world.llm_world_prompt_builder import LLMWorldPromptBuilder


def test_build_messages_contains_world_safety_contract_and_payload() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Mock cloud IAM advisory",
        summary="A mock advisory about cloud IAM risk.",
        category=WorldEventCategory.CYBERSECURITY,
        source_name="Mock Cyber Feed",
        tags=["cloud", "iam"],
    )
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        event_id="event-1",
        title="Review security",
        message="Review IAM controls.",
    )

    messages = LLMWorldPromptBuilder().build_messages(
        briefing_type="world_briefing",
        events=[event],
        suggestions=[suggestion],
        alerts=[suggestion],
        context={"interests": ["Jarvis project"]},
    )

    assert [message.role for message in messages] == [LLMRole.SYSTEM, LLMRole.USER]
    system = messages[0].content
    user = messages[1].content
    assert "do not fetch live data" in system.lower()
    assert "do not invent events" in system.lower()
    assert "use only the provided events" in system.lower()
    assert "Return JSON only" in system
    assert "mock or placeholder" in system
    assert "Mock cloud IAM advisory" in user
    assert "Review security" in user
    assert "Jarvis project" in user
