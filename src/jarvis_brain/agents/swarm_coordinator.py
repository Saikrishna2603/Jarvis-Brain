from jarvis_brain.agents.swarm_agent import SwarmProposal
from jarvis_brain.agents.swarm_runtime import SwarmRuntime
from jarvis_platform.safety.permission_policy import PermissionPolicyEngine
from jarvis_platform.safety.risk_classifier import RiskClassifier
from jarvis_platform.schemas.common import RiskLevel
from jarvis_platform.security.action_firewall import ActionFirewall


class ExecutionReviewAgent:
    """Validate swarm proposals before any future execution path."""

    name = "execution_review_agent"

    def __init__(
        self,
        risk_classifier: RiskClassifier | None = None,
        permission_policy: PermissionPolicyEngine | None = None,
        action_firewall: ActionFirewall | None = None,
    ) -> None:
        """Create the execution reviewer."""
        self.risk_classifier = risk_classifier or RiskClassifier()
        self.permission_policy = permission_policy or PermissionPolicyEngine()
        self.action_firewall = action_firewall or ActionFirewall()

    def review(self, proposal: SwarmProposal) -> SwarmProposal:
        """Apply safety checks to one proposal."""
        firewall = self.action_firewall.allow_action(
            proposal.proposed_action,
            target=proposal.target,
            metadata=proposal.metadata,
        )
        risk = self.risk_classifier.classify_action(
            proposal.proposed_action,
            target=proposal.target,
        )
        requires_approval = self.permission_policy.should_require_approval(
            proposal.proposed_action,
            proposal.target,
            risk,
        )

        proposal.risk_level = risk.level.value
        proposal.requires_approval = bool(
            requires_approval or firewall["requires_approval"]
        )
        proposal.metadata["firewall_reason"] = firewall["reason"]
        proposal.metadata["risk_reasons"] = risk.reasons

        if not firewall["allowed"] or risk.level == RiskLevel.BLOCKED:
            proposal.rejected = True
            proposal.rejection_reason = firewall["reason"]

        return proposal

    def review_many(self, proposals: list[SwarmProposal]) -> list[SwarmProposal]:
        """Review a list of proposals."""
        return [self.review(proposal) for proposal in proposals]


class SwarmCoordinator:
    """Coordinate specialist swarm agents safely."""

    def __init__(
        self,
        runtime: SwarmRuntime | None = None,
        execution_review_agent: ExecutionReviewAgent | None = None,
    ) -> None:
        """Create a coordinator with a proposal runtime and reviewer."""
        self.runtime = runtime or SwarmRuntime()
        self.execution_review_agent = execution_review_agent or ExecutionReviewAgent()

    def preview(self, raw_input: str, context: dict | None = None) -> dict:
        """Return a reviewed multi-agent preview without execution."""
        proposals = self.runtime.run_preview(raw_input, context=context)
        reviewed = self.execution_review_agent.review_many(proposals)
        return self._result(
            raw_input=raw_input,
            proposals=reviewed,
            mode="preview",
        )

    def run_safe(self, raw_input: str, context: dict | None = None) -> dict:
        """Run the safe swarm path.

        This still does not execute tools; it only returns reviewed proposals.
        """
        proposals = self.runtime.run_preview(raw_input, context=context)
        reviewed = self.execution_review_agent.review_many(proposals)
        return self._result(
            raw_input=raw_input,
            proposals=reviewed,
            mode="run_safe",
        )

    def status(self) -> dict:
        """Return coordinator status."""
        runtime_status = self.runtime.status()
        return {
            **runtime_status,
            "coordinator": "swarm_coordinator",
            "execution_review_agent": self.execution_review_agent.name,
            "executes_tools": False,
            "requires_safety_review": True,
        }

    def _result(
        self,
        raw_input: str,
        proposals: list[SwarmProposal],
        mode: str,
    ) -> dict:
        accepted = [proposal for proposal in proposals if not proposal.rejected]
        rejected = [proposal for proposal in proposals if proposal.rejected]
        return {
            "status": "success",
            "mode": mode,
            "raw_input": raw_input,
            "executed": False,
            "agent_count": len(proposals),
            "proposals": [proposal.to_dict() for proposal in proposals],
            "accepted_count": len(accepted),
            "rejected_count": len(rejected),
            "final_review": {
                "reviewed_by": self.execution_review_agent.name,
                "safe_to_execute_without_approval": all(
                    not proposal.requires_approval and not proposal.rejected
                    for proposal in proposals
                ),
                "approval_required": any(
                    proposal.requires_approval for proposal in proposals
                ),
            },
        }
