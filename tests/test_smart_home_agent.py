import pytest

from jarvis_brain.agents.smart_home_agent import SmartHomeAgent


def test_smart_home_agent_has_correct_name_and_domain() -> None:
    agent = SmartHomeAgent()

    assert agent.name == "smart_home_agent"
    assert agent.domain == "smart_home"


def test_smart_home_agent_supported_actions_is_not_empty() -> None:
    agent = SmartHomeAgent()

    assert agent.supported_actions


def test_smart_home_agent_can_handle_supported_action() -> None:
    agent = SmartHomeAgent()

    assert agent.can_handle("turn_off_light") is True


def test_smart_home_agent_can_handle_returns_false_for_unsupported_action() -> None:
    agent = SmartHomeAgent()

    assert agent.can_handle("summarize_spending") is False


@pytest.mark.parametrize(
    "action",
    [
        "list_devices",
        "turn_on_light",
        "turn_off_light",
        "set_temperature",
        "lock_door",
        "unlock_door",
    ],
)
def test_smart_home_agent_handle_works_for_supported_actions(action: str) -> None:
    agent = SmartHomeAgent()

    result = agent.handle(action=action, target="living room", payload={"source": "test"})

    assert result["status"] == "success"
    assert result["agent"] == "smart_home_agent"
    assert result["domain"] == "smart_home"
    assert result["action"] == action
    assert result["target"] == "living room"
    assert result["payload"] == {"source": "test"}


def test_smart_home_agent_unsupported_action_raises_value_error() -> None:
    agent = SmartHomeAgent()

    with pytest.raises(ValueError, match="SmartHomeAgent cannot handle action: unknown"):
        agent.handle("unknown")
