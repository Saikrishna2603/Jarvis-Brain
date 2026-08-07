from jarvis_platform.schemas.brain_orchestration import BrainOrchestratorProposal
from jarvis_platform.security.action_firewall import ActionFirewall
from jarvis_platform.security.secret_policy import SecretPolicyEngine


class BrainOrchestrationValidator:
    """Deterministically validate LLM-proposed orchestration output."""

    blocked_phrases = (
        "bypass security",
        "disable safety",
        "reveal secret",
        "exfiltrate",
        "steal credentials",
        "steal passwords",
        "malware",
    )

    def __init__(
        self,
        action_firewall: ActionFirewall | None = None,
        secret_policy: SecretPolicyEngine | None = None,
        minimum_confidence: float = 0.55,
    ) -> None:
        self.action_firewall = action_firewall or ActionFirewall()
        self.secret_policy = secret_policy or SecretPolicyEngine()
        self.minimum_confidence = minimum_confidence

    def validate(self, proposal: BrainOrchestratorProposal) -> tuple[bool, str]:
        if proposal.confidence < self.minimum_confidence:
            return False, "LLM orchestration confidence was below threshold."
        combined_text = " ".join(
            [
                proposal.summary,
                proposal.response_strategy,
                *[step.title for step in proposal.plan_steps],
                *[step.description for step in proposal.plan_steps],
                *[tool.action for tool in proposal.tool_proposals],
            ]
        ).lower()
        if any(phrase in combined_text for phrase in self.blocked_phrases):
            return False, "LLM proposal contained blocked or unsafe language."
        secret_scan = self.secret_policy.inspect_text(combined_text, context="brain_orchestration")
        if secret_scan.has_secrets:
            return False, "LLM proposal exposed sensitive data."
        for tool in proposal.tool_proposals:
            firewall = self.action_firewall.allow_action(tool.action, tool.target)
            if not bool(firewall["allowed"]):
                return False, str(firewall["reason"])
        return True, "LLM orchestration proposal accepted."

