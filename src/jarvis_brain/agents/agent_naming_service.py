from jarvis_platform.schemas.agent_lifecycle import AgentRole


class AgentNamingService:
    """Generate deterministic role-aware names for Jarvis agents."""

    NAME_POOLS: dict[AgentRole, list[str]] = {
        AgentRole.PLANNER: ["Forge", "Architect", "Mapmaker"],
        AgentRole.RESEARCHER: ["Scout", "Atlas", "Horizon"],
        AgentRole.CODER: ["Vector", "Syntax", "Builder"],
        AgentRole.SECURITY: ["Sentinel", "Shield", "Cipher"],
        AgentRole.MEMORY: ["Archive", "Memo", "Helix"],
        AgentRole.WORLD: ["Beacon", "Orbit", "Horizon"],
        AgentRole.VISION: ["Iris", "Lens", "Optic"],
        AgentRole.VOICE: ["Echo", "Vox", "Resonance"],
        AgentRole.INTEGRATION: ["Relay", "Bridge", "Link"],
        AgentRole.REVIEWER: ["Auditor", "Prism", "Lens"],
        AgentRole.EXECUTOR: ["Pulse", "Operator", "Relay"],
        AgentRole.UNKNOWN: ["Nova", "Unit", "Node"],
    }

    def generate_name(self, role: AgentRole, sequence: int) -> str:
        """Return a deterministic name such as Forge-01."""
        return f"{self.get_base_name(role, sequence)}-{sequence:02d}"

    def get_base_name(self, role: AgentRole, sequence: int) -> str:
        """Return the deterministic role-aware base name for a sequence."""
        if sequence < 1:
            raise ValueError("Agent name sequence must be at least 1.")
        names = self.NAME_POOLS.get(role, self.NAME_POOLS[AgentRole.UNKNOWN])
        return names[(sequence - 1) % len(names)]
