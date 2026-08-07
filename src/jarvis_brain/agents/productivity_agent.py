class ProductivityAgent:
    """Mock specialist agent for productivity planning and interpretation.

    This agent does not read files, send email, or create calendar events. It
    only prepares domain-aware mock results for productivity actions.
    """

    name = "productivity_agent"
    domain = "productivity"
    supported_actions = [
        "list_events",
        "create_event",
        "draft_email",
        "send_email",
        "list_files",
        "read_file",
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
        """Prepare a mock productivity result for a supported action."""
        if not self.can_handle(action):
            raise ValueError(f"ProductivityAgent cannot handle action: {action}")

        return {
            "status": "success",
            "agent": self.name,
            "domain": self.domain,
            "action": action,
            "target": target,
            "payload": payload or {},
            "message": f"Mock productivity agent prepared action: {action}",
        }
