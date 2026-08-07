from typing import Any


class BrainContextBuilder:
    """Build a compact, safe context bundle for orchestration.

    This foundation intentionally accepts optional managers so it can be wired
    into the existing modular monolith without creating import cycles.
    """

    def __init__(
        self,
        context_manager: Any | None = None,
        semantic_memory_manager: Any | None = None,
        agent_lifecycle_manager: Any | None = None,
        token_budget: int = 3000,
    ) -> None:
        self.context_manager = context_manager
        self.semantic_memory_manager = semantic_memory_manager
        self.agent_lifecycle_manager = agent_lifecycle_manager
        self.token_budget = token_budget

    def build(self, raw_input: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
        """Return safe, bounded orchestration context."""
        metadata = dict(metadata or {})
        context: dict[str, Any] = {
            "request_source": metadata.get("source", "text"),
            "token_budget": self.token_budget,
            "available_context": [],
            "active_agents": [],
            "project_state": metadata.get("project_state", {}),
            "system_state": metadata.get("system_state", {}),
        }
        if self.agent_lifecycle_manager is not None:
            try:
                snapshot = self.agent_lifecycle_manager.get_snapshot()
                context["active_agents"] = [
                    {
                        "agent_id": agent.agent_id,
                        "name": agent.name,
                        "role": agent.role.value,
                        "status": agent.status.value,
                    }
                    for agent in snapshot.active_agents[:8]
                ]
            except Exception:
                context["active_agents_unavailable"] = True
        if self.semantic_memory_manager is not None and metadata.get("include_memory_preview"):
            context["memory_hint"] = "Semantic memory is available; query generation may be proposed."
        context["input_preview"] = raw_input[:500]
        return context

