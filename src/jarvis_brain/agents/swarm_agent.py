from dataclasses import dataclass, field
from uuid import uuid4

from jarvis_platform.schemas.common import utc_now


@dataclass
class SwarmProposal:
    """A non-executing proposal from a swarm agent."""

    agent_name: str
    title: str
    summary: str
    proposed_action: str = "summarize_information"
    target: str | None = None
    risk_level: str = "low"
    requires_approval: bool = False
    rejected: bool = False
    rejection_reason: str | None = None
    metadata: dict = field(default_factory=dict)
    proposal_id: str = field(default_factory=lambda: str(uuid4()))

    def to_dict(self) -> dict:
        """Return a JSON-safe proposal dictionary."""
        return {
            "proposal_id": self.proposal_id,
            "agent_name": self.agent_name,
            "title": self.title,
            "summary": self.summary,
            "proposed_action": self.proposed_action,
            "target": self.target,
            "risk_level": self.risk_level,
            "requires_approval": self.requires_approval,
            "rejected": self.rejected,
            "rejection_reason": self.rejection_reason,
            "metadata": self.metadata,
        }


class SwarmAgent:
    """Base class for proposal-only specialist swarm agents."""

    name = "swarm_agent"
    specialty = "general"

    def can_handle(self, raw_input: str) -> bool:
        """Return True when this agent should contribute."""
        return True

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        """Return a safe generic proposal."""
        return SwarmProposal(
            agent_name=self.name,
            title="General analysis",
            summary="Review the request and keep all outputs proposal-only.",
            metadata={"created_at": utc_now().isoformat()},
        )


class PlannerSwarmAgent(SwarmAgent):
    """Agent that proposes a safe task breakdown."""

    name = "planner_agent"
    specialty = "planning"

    def can_handle(self, raw_input: str) -> bool:
        text = raw_input.lower()
        return any(word in text for word in ["plan", "build", "phase", "steps", "design"])

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        return SwarmProposal(
            agent_name=self.name,
            title="Planning breakdown",
            summary="Break the request into scoped steps, then review each proposed step before execution.",
            proposed_action="create_plan",
            target=raw_input,
            risk_level="low",
        )


class ResearchSwarmAgent(SwarmAgent):
    """Agent that proposes safe evidence gathering."""

    name = "research_agent"
    specialty = "research"

    def can_handle(self, raw_input: str) -> bool:
        text = raw_input.lower()
        return any(word in text for word in ["research", "source", "evidence", "docs", "learn"])

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        return SwarmProposal(
            agent_name=self.name,
            title="Research pass",
            summary="Gather trusted source context and mark unverified information before using it.",
            proposed_action="retrieve_evidence",
            target=raw_input,
            risk_level="low",
        )


class CodingSwarmAgent(SwarmAgent):
    """Agent that proposes code-review or implementation analysis."""

    name = "coding_agent"
    specialty = "coding"

    def can_handle(self, raw_input: str) -> bool:
        text = raw_input.lower()
        return any(
            word in text
            for word in ["code", "python", "fastapi", "bug", "test", "repo", "function", "class"]
        )

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        return SwarmProposal(
            agent_name=self.name,
            title="Coding analysis",
            summary="Inspect code context, identify likely changes, and keep execution behind review.",
            proposed_action="inspect_code",
            target=raw_input,
            risk_level="low",
        )


class SecuritySwarmAgent(SwarmAgent):
    """Agent that proposes security review and refuses unsafe goals."""

    name = "security_agent"
    specialty = "security"

    def can_handle(self, raw_input: str) -> bool:
        text = raw_input.lower()
        return any(
            word in text
            for word in ["security", "credential", "password", "secret", "malware", "exploit", "phishing"]
        )

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        text = raw_input.lower()
        unsafe = any(
            phrase in text
            for phrase in [
                "steal credentials",
                "steal password",
                "malware",
                "bypass security",
                "exfiltrate",
                "phishing kit",
            ]
        )
        if unsafe:
            return SwarmProposal(
                agent_name=self.name,
                title="Unsafe security request refused",
                summary="Refuse offensive or credential-theft assistance and redirect to defensive security guidance.",
                proposed_action="bypass_security",
                target=raw_input,
                risk_level="blocked",
                rejected=True,
                rejection_reason="Unsafe cybersecurity request.",
            )
        return SwarmProposal(
            agent_name=self.name,
            title="Security review",
            summary="Check proposed work for credential exposure, unsafe actions, and approval requirements.",
            proposed_action="review_security",
            target=raw_input,
            risk_level="medium",
            requires_approval=False,
        )


class MemorySwarmAgent(SwarmAgent):
    """Agent that proposes memory lookup without storing sensitive data."""

    name = "memory_agent"
    specialty = "memory"

    def can_handle(self, raw_input: str) -> bool:
        text = raw_input.lower()
        return any(word in text for word in ["remember", "memory", "project", "context"])

    def propose(self, raw_input: str, context: dict | None = None) -> SwarmProposal:
        return SwarmProposal(
            agent_name=self.name,
            title="Memory context",
            summary="Use safe project memory context if available, without storing secrets automatically.",
            proposed_action="summarize_information",
            target="semantic_memory",
            risk_level="low",
        )
