from jarvis_brain.agents.agent_message_bus import AgentMessage, AgentMessageBus
from jarvis_brain.agents.swarm_agent import (
    CodingSwarmAgent,
    MemorySwarmAgent,
    PlannerSwarmAgent,
    ResearchSwarmAgent,
    SecuritySwarmAgent,
    SwarmAgent,
    SwarmProposal,
)


class SwarmRuntime:
    """Run proposal-only specialist agents for a request."""

    def __init__(
        self,
        agents: list[SwarmAgent] | None = None,
        message_bus: AgentMessageBus | None = None,
    ) -> None:
        """Create a runtime with default specialist agents."""
        self.agents = agents or [
            PlannerSwarmAgent(),
            ResearchSwarmAgent(),
            CodingSwarmAgent(),
            SecuritySwarmAgent(),
            MemorySwarmAgent(),
        ]
        self.message_bus = message_bus or AgentMessageBus()

    def run_preview(
        self,
        raw_input: str,
        context: dict | None = None,
    ) -> list[SwarmProposal]:
        """Collect non-executing proposals from relevant agents."""
        proposals: list[SwarmProposal] = []
        for agent in self.agents:
            if agent.can_handle(raw_input):
                proposal = agent.propose(raw_input, context=context)
                proposals.append(proposal)
                self.message_bus.publish(
                    AgentMessage(
                        sender=agent.name,
                        recipient="execution_review_agent",
                        content=proposal.summary,
                        metadata=proposal.to_dict(),
                    )
                )

        if not proposals:
            fallback = PlannerSwarmAgent().propose(raw_input, context=context)
            proposals.append(fallback)
            self.message_bus.publish(
                AgentMessage(
                    sender=fallback.agent_name,
                    recipient="execution_review_agent",
                    content=fallback.summary,
                    metadata=fallback.to_dict(),
                )
            )

        return proposals

    def status(self) -> dict:
        """Return a safe runtime status."""
        return {
            "status": "ready",
            "agent_count": len(self.agents),
            "agents": [agent.name for agent in self.agents],
            "proposal_only": True,
            "message_count": len(self.message_bus.all_messages()),
        }
