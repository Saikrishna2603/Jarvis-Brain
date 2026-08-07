import pytest

from jarvis_brain.agents.agent_interface import BaseAgent


class FakeAgent:
    """Simple test agent for the agent interface contract."""

    name = "fake_agent"
    domain = "testing"
    supported_actions = ["fake_action"]

    def can_handle(self, action: str) -> bool:
        """Return True for actions supported by this fake agent."""
        return action in self.supported_actions

    def handle(
        self,
        action: str,
        target: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Return a fake successful result for supported actions."""
        if not self.can_handle(action):
            raise ValueError(f"FakeAgent cannot handle action: {action}")

        return {
            "status": "success",
            "agent": self.name,
            "domain": self.domain,
            "action": action,
            "target": target,
            "payload": payload or {},
        }


def test_fake_agent_can_be_created() -> None:
    agent: BaseAgent = FakeAgent()

    assert agent.name == "fake_agent"
    assert agent.domain == "testing"
    assert agent.supported_actions == ["fake_action"]


def test_can_handle_returns_true_for_supported_action() -> None:
    agent = FakeAgent()

    assert agent.can_handle("fake_action") is True


def test_can_handle_returns_false_for_unsupported_action() -> None:
    agent = FakeAgent()

    assert agent.can_handle("unknown_action") is False


def test_handle_returns_expected_result() -> None:
    agent = FakeAgent()

    result = agent.handle(
        action="fake_action",
        target="target",
        payload={"key": "value"},
    )

    assert result == {
        "status": "success",
        "agent": "fake_agent",
        "domain": "testing",
        "action": "fake_action",
        "target": "target",
        "payload": {"key": "value"},
    }


def test_unsupported_action_raises_value_error() -> None:
    agent = FakeAgent()

    with pytest.raises(ValueError, match="FakeAgent cannot handle action: unknown_action"):
        agent.handle("unknown_action")
