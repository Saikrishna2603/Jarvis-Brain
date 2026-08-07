from typing import Protocol


class BaseAgent(Protocol):
    """Shared interface for Jarvis domain agents.

    Agents are higher-level reasoning or coordination units. They decide how to
    handle domain actions, while drivers remain responsible for specific tools
    and external services.
    """

    name: str
    domain: str
    supported_actions: list[str]

    def can_handle(self, action: str) -> bool:
        """Return True when this agent supports the action."""
        ...

    def handle(
        self,
        action: str,
        target: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Handle an action and return a structured result."""
        ...
