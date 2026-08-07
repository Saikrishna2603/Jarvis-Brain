from jarvis_platform.schemas.brain_orchestration import BrainAgentTeamProposal, BrainIntentType


class AgentTeamBuilder:
    """Validate and normalize model-proposed specialist teams."""

    allowed_roles = {
        "planner",
        "researcher",
        "coder",
        "security",
        "memory",
        "world",
        "vision",
        "voice",
        "integration",
        "reviewer",
        "executor",
        "reflection",
    }

    def propose_from_intents(self, intents: list[BrainIntentType]) -> BrainAgentTeamProposal:
        roles: list[str] = ["planner"]
        if BrainIntentType.CODING in intents:
            roles.extend(["coder", "reviewer"])
        if BrainIntentType.RESEARCH in intents:
            roles.append("researcher")
        if BrainIntentType.WORLD in intents:
            roles.append("world")
        if BrainIntentType.MEMORY in intents:
            roles.append("memory")
        if BrainIntentType.VISION in intents:
            roles.append("vision")
        if BrainIntentType.VOICE in intents:
            roles.append("voice")
        if BrainIntentType.EXECUTION in intents or BrainIntentType.AUTOMATION in intents:
            roles.extend(["security", "executor"])
        return BrainAgentTeamProposal(
            roles=self._dedupe_allowed(roles),
            reason="Specialists selected from structured intent categories.",
        )

    def validate(self, proposal: BrainAgentTeamProposal) -> BrainAgentTeamProposal:
        return proposal.model_copy(update={"roles": self._dedupe_allowed(proposal.roles)})

    def _dedupe_allowed(self, roles: list[str]) -> list[str]:
        result: list[str] = []
        for role in roles:
            normalized = role.strip().lower()
            if normalized in self.allowed_roles and normalized not in result:
                result.append(normalized)
        return result

