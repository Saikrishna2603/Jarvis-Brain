import pytest

from jarvis_brain.agents.finance_agent import FinanceAgent


def test_finance_agent_has_correct_name_and_domain() -> None:
    agent = FinanceAgent()

    assert agent.name == "finance_agent"
    assert agent.domain == "finance"


def test_finance_agent_supported_actions_is_not_empty() -> None:
    agent = FinanceAgent()

    assert agent.supported_actions


def test_finance_agent_can_handle_supported_action() -> None:
    agent = FinanceAgent()

    assert agent.can_handle("summarize_spending") is True


def test_finance_agent_can_handle_returns_false_for_unsupported_action() -> None:
    agent = FinanceAgent()

    assert agent.can_handle("turn_off_light") is False


@pytest.mark.parametrize(
    "action",
    [
        "list_accounts",
        "summarize_spending",
        "detect_subscriptions",
        "prepare_payment",
        "execute_payment",
    ],
)
def test_finance_agent_handle_works_for_supported_actions(action: str) -> None:
    agent = FinanceAgent()

    result = agent.handle(action=action, target="test target", payload={"source": "test"})

    assert result["status"] == "success"
    assert result["agent"] == "finance_agent"
    assert result["domain"] == "finance"
    assert result["action"] == action
    assert result["target"] == "test target"
    assert result["payload"] == {"source": "test"}


def test_finance_agent_unsupported_action_raises_value_error() -> None:
    agent = FinanceAgent()

    with pytest.raises(ValueError, match="FinanceAgent cannot handle action: unknown"):
        agent.handle("unknown")
