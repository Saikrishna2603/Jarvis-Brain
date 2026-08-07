from jarvis_brain.agents.agent_interface import BaseAgent
from jarvis_brain.agents.finance_agent import FinanceAgent
from jarvis_brain.agents.productivity_agent import ProductivityAgent
from jarvis_brain.agents.smart_home_agent import SmartHomeAgent


class AgentRuntime:
    """Registry and dispatcher for Jarvis specialist agents."""

    def __init__(self) -> None:
        """Create an empty agent runtime."""
        self._agents: list[BaseAgent] = []

    def register_agent(self, agent: BaseAgent) -> None:
        """Register an agent that can handle one or more actions."""
        self._agents.append(agent)

    def get_agent_for_action(self, action: str) -> BaseAgent:
        """Return the first registered agent that supports an action."""
        for agent in self._agents:
            if agent.can_handle(action):
                return agent

        raise ValueError(f"No registered agent can handle action: {action}")

    def list_agents(self) -> list[BaseAgent]:
        """Return all registered agents."""
        return list(self._agents)

    def list_supported_actions(self) -> list[str]:
        """Return every action supported by registered agents."""
        actions: list[str] = []
        for agent in self._agents:
            for action in agent.supported_actions:
                if action not in actions:
                    actions.append(action)

        return actions

    def handle_action(
        self,
        action: str,
        target: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Handle an action through the matching specialist agent."""
        agent = self.get_agent_for_action(action)
        return agent.handle(action=action, target=target, payload=payload)


def create_default_agent_runtime() -> AgentRuntime:
    """Create an agent runtime with all default Jarvis v1 agents."""
    runtime = AgentRuntime()
    runtime.register_agent(FinanceAgent())
    runtime.register_agent(SmartHomeAgent())
    runtime.register_agent(ProductivityAgent())
    return runtime
