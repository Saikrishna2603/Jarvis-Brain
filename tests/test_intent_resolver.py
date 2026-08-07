from jarvis_brain.engine.intent_resolver import IntentResolver
import pytest


def test_resolves_youtube_command() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("Hey Jarvis YouTube")

    assert intent.intent_type == "action"
    assert intent.name == "action"
    assert intent.action == "open_website"
    assert intent.target == "YouTube"
    assert intent.confidence == 0.95
    assert intent.raw_input == "Hey Jarvis YouTube"


def test_resolves_approval_confirm() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("go ahead")

    assert intent.intent_type == "approval_confirm"
    assert intent.confidence == 0.95


def test_resolves_approval_reject() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("stop")

    assert intent.intent_type == "approval_reject"
    assert intent.confidence == 0.95


def test_resolves_turn_off_lights() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("turn off living room lights")

    assert intent.intent_type == "action"
    assert intent.action == "turn_off_light"
    assert intent.target == "living room lights"


def test_resolves_send_email() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("send email")

    assert intent.intent_type == "action"
    assert intent.action == "send_email"
    assert intent.target == "email"


def test_resolves_summarize_spending() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("summarize spending")

    assert intent.intent_type == "action"
    assert intent.action == "summarize_spending"
    assert intent.target is None


def test_resolves_pay_bill() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("pay bill")

    assert intent.intent_type == "action"
    assert intent.action == "execute_payment"
    assert intent.target == "bill"


def test_resolves_review_my_finances_as_goal_intent() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("review my finances")

    assert intent.intent_type == "goal"
    assert intent.goal == "review my finances"
    assert intent.requires_plan is True
    assert intent.confidence == 0.95


def test_resolves_secure_my_home_as_goal_intent() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("secure my home")

    assert intent.intent_type == "goal"
    assert intent.goal == "secure my home"
    assert intent.requires_plan is True


def test_unknown_input_returns_unknown_intent() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("what is the moon made of")

    assert intent.intent_type == "unknown"
    assert intent.name == "unknown"
    assert intent.confidence == 0.0
    assert intent.needs_clarification is True


def test_partial_keyword_match_has_lower_confidence() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("could you please open youtube for me")

    assert intent.intent_type == "action"
    assert intent.action == "open_website"
    assert intent.target == "YouTube"
    assert intent.confidence == 0.75


def test_intent_result_serializes_correctly() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("review my finances")
    dumped = intent.model_dump()
    json_text = intent.model_dump_json()

    assert dumped["intent_type"] == "goal"
    assert dumped["goal"] == "review my finances"
    assert dumped["metadata"] == {"goal": "review my finances"}
    assert isinstance(json_text, str)


@pytest.mark.parametrize(
    "raw_input",
    [
        "world briefing",
        "give me a world briefing",
        "jarvis give me a world briefing",
        "what is happening in the world",
        "global briefing",
        "global updates",
        "world update",
        "world updates",
        "give me world updates",
        "give me global updates",
        "show what's happening in the world today",
        "show what is happening in the world today",
        "what's happening in the world today",
        "what is happening in the world today",
        "what's going on in the world today",
        "what is going on in the world today",
        "show me what's going on in the world",
        "show me what is going on in the world",
        "what happened in the world today",
        "show world news today",
        "today's world briefing",
        "daily world briefing",
        "world update today",
    ],
)
def test_resolves_world_briefing_intent(raw_input: str) -> None:
    resolver = IntentResolver()

    intent = resolver.resolve(raw_input)

    assert intent.intent_type == "world_intelligence"
    assert intent.action == "get_world_briefing"
    assert intent.target == "global"
    assert intent.confidence >= 0.9


@pytest.mark.parametrize(
    "raw_input",
    [
        "any cyber alerts today",
        "show cyber alerts",
        "security alerts",
        "cybersecurity updates",
        "cloud security alerts",
    ],
)
def test_resolves_cyber_alerts_intent(raw_input: str) -> None:
    resolver = IntentResolver()

    intent = resolver.resolve(raw_input)

    assert intent.intent_type == "world_intelligence"
    assert intent.action == "get_cyber_alerts"
    assert intent.target == "cybersecurity"


def test_resolves_project_relevant_updates_intent() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("what global updates matter to my project")

    assert intent.intent_type == "world_intelligence"
    assert intent.action == "get_project_relevant_updates"
    assert intent.target == "Jarvis project"


def test_resolves_ai_research_updates_intent() -> None:
    resolver = IntentResolver()

    intent = resolver.resolve("any ai research updates")

    assert intent.intent_type == "world_intelligence"
    assert intent.action == "get_ai_research_updates"
    assert intent.target == "ai_research"


@pytest.mark.parametrize(
    "raw_input",
    [
        "show me world alerts",
        "any important alerts",
        "urgent world alerts",
        "world alerts",
    ],
)
def test_resolves_world_alerts_intent(raw_input: str) -> None:
    resolver = IntentResolver()

    intent = resolver.resolve(raw_input)

    assert intent.intent_type == "world_intelligence"
    assert intent.action == "get_world_alerts"
    assert intent.target == "alerts"
