import pytest

from jarvis_brain.agents.productivity_agent import ProductivityAgent


def test_productivity_agent_has_correct_name_and_domain() -> None:
    agent = ProductivityAgent()

    assert agent.name == "productivity_agent"
    assert agent.domain == "productivity"


def test_productivity_agent_supported_actions_is_not_empty() -> None:
    agent = ProductivityAgent()

    assert agent.supported_actions


def test_productivity_agent_can_handle_supported_action() -> None:
    agent = ProductivityAgent()

    assert agent.can_handle("draft_email") is True


def test_productivity_agent_can_handle_returns_false_for_unsupported_action() -> None:
    agent = ProductivityAgent()

    assert agent.can_handle("unlock_door") is False


@pytest.mark.parametrize(
    "action",
    [
        "list_events",
        "create_event",
        "draft_email",
        "send_email",
        "list_files",
        "read_file",
    ],
)
def test_productivity_agent_handle_works_for_supported_actions(action: str) -> None:
    agent = ProductivityAgent()

    result = agent.handle(action=action, target="test target", payload={"source": "test"})

    assert result["status"] == "success"
    assert result["agent"] == "productivity_agent"
    assert result["domain"] == "productivity"
    assert result["action"] == action
    assert result["target"] == "test target"
    assert result["payload"] == {"source": "test"}


def test_productivity_agent_unsupported_action_raises_value_error() -> None:
    agent = ProductivityAgent()

    with pytest.raises(ValueError, match="ProductivityAgent cannot handle action: unknown"):
        agent.handle("unknown")
