class FinanceAgent:
    """Mock specialist agent for finance planning and interpretation.

    This agent does not connect to banks, payment systems, or finance APIs. It
    only prepares domain-aware mock results for finance-related actions.
    """

    name = "finance_agent"
    domain = "finance"
    supported_actions = [
        "list_accounts",
        "summarize_spending",
        "detect_subscriptions",
        "prepare_payment",
        "execute_payment",
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
        """Prepare a mock finance result for a supported action."""
        if not self.can_handle(action):
            raise ValueError(f"FinanceAgent cannot handle action: {action}")

        return {
            "status": "success",
            "agent": self.name,
            "domain": self.domain,
            "action": action,
            "target": target,
            "payload": payload or {},
            "message": f"Mock finance agent prepared action: {action}",
        }
