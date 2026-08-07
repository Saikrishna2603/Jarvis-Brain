import pytest

from jarvis_brain.agents.agent_runtime import AgentRuntime, create_default_agent_runtime
from jarvis_brain.agents.finance_agent import FinanceAgent
from jarvis_brain.agents.productivity_agent import ProductivityAgent
from jarvis_brain.agents.smart_home_agent import SmartHomeAgent


def test_agent_runtime_can_register_agents() -> None:
    runtime = AgentRuntime()
    agent = FinanceAgent()

    runtime.register_agent(agent)

    assert runtime.list_agents() == [agent]


def test_can_find_finance_agent_for_summarize_spending() -> None:
    runtime = create_default_agent_runtime()

    agent = runtime.get_agent_for_action("summarize_spending")

    assert isinstance(agent, FinanceAgent)
    assert agent.name == "finance_agent"


def test_can_find_smart_home_agent_for_turn_off_light() -> None:
    runtime = create_default_agent_runtime()

    agent = runtime.get_agent_for_action("turn_off_light")

    assert isinstance(agent, SmartHomeAgent)
    assert agent.name == "smart_home_agent"


def test_can_find_productivity_agent_for_draft_email() -> None:
    runtime = create_default_agent_runtime()

    agent = runtime.get_agent_for_action("draft_email")

    assert isinstance(agent, ProductivityAgent)
    assert agent.name == "productivity_agent"


def test_list_agents_returns_registered_agents() -> None:
    runtime = create_default_agent_runtime()

    agents = runtime.list_agents()

    assert len(agents) == 3
    assert [agent.name for agent in agents] == [
        "finance_agent",
        "smart_home_agent",
        "productivity_agent",
    ]


def test_list_supported_actions_returns_actions() -> None:
    runtime = create_default_agent_runtime()

    actions = runtime.list_supported_actions()

    assert "summarize_spending" in actions
    assert "turn_off_light" in actions
    assert "draft_email" in actions


def test_handle_action_routes_to_correct_agent() -> None:
    runtime = create_default_agent_runtime()

    result = runtime.handle_action(
        action="turn_off_light",
        target="living room lights",
        payload={"source": "test"},
    )

    assert result["status"] == "success"
    assert result["agent"] == "smart_home_agent"
    assert result["domain"] == "smart_home"
    assert result["action"] == "turn_off_light"
    assert result["target"] == "living room lights"
    assert result["payload"] == {"source": "test"}


def test_unknown_action_raises_value_error() -> None:
    runtime = create_default_agent_runtime()

    with pytest.raises(
        ValueError,
        match="No registered agent can handle action: unknown_action",
    ):
        runtime.get_agent_for_action("unknown_action")


def test_handle_unknown_action_raises_value_error() -> None:
    runtime = create_default_agent_runtime()

    with pytest.raises(
        ValueError,
        match="No registered agent can handle action: unknown_action",
    ):
        runtime.handle_action("unknown_action")


def test_create_default_agent_runtime_registers_all_default_agents() -> None:
    runtime = create_default_agent_runtime()

    agent_types = {type(agent) for agent in runtime.list_agents()}

    assert agent_types == {FinanceAgent, SmartHomeAgent, ProductivityAgent}
