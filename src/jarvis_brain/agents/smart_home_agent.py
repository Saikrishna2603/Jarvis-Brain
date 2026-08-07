class SmartHomeAgent:
    """Mock specialist agent for smart home planning and interpretation.

    This agent does not control real devices. It only prepares domain-aware mock
    results for smart home actions.
    """

    name = "smart_home_agent"
    domain = "smart_home"
    supported_actions = [
        "list_devices",
        "turn_on_light",
        "turn_off_light",
        "set_temperature",
        "lock_door",
        "unlock_door",
    ]

    def can_handle(self, action: str) -> bool:
        """Return True when this agent supports the action."""
        return action in self.supported_actions

    def handle(
        self,
        action: str,
        target: str | None = None,
        payload: dict | None = None,
    ) -> dict:
        """Prepare a mock smart home result for a supported action."""
        if not self.can_handle(action):
            raise ValueError(f"SmartHomeAgent cannot handle action: {action}")

        return {
            "status": "success",
            "agent": self.name,
            "domain": self.domain,
            "action": action,
            "target": target,
            "payload": payload or {},
            "message": f"Mock smart home agent prepared action: {action}",
        }
