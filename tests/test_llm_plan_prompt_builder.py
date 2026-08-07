from jarvis_brain.engine.llm_plan_prompt_builder import LLMPlanPromptBuilder
from jarvis_platform.schemas.llm import LLMRole


def test_build_messages_contains_safe_planning_contract() -> None:
    messages = LLMPlanPromptBuilder().build_messages(
        "Prepare Jarvis for Gmail integration safely",
        context={"project": "Jarvis"},
    )

    assert [message.role for message in messages] == [LLMRole.SYSTEM, LLMRole.USER]
    assert "only proposing a plan" in messages[0].content
    assert "do not execute tools" in messages[0].content
    assert "Return JSON only" in messages[0].content
    assert "inspect_code" in messages[0].content
    assert "execute_shell" in messages[0].content
    assert "Prepare Jarvis for Gmail integration safely" in messages[1].content
    assert '"project": "Jarvis"' in messages[1].content
