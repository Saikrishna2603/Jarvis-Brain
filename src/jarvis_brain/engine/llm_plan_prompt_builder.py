import json

from jarvis_platform.schemas.llm import LLMMessage, LLMRole


class LLMPlanPromptBuilder:
    """Build a constrained, planning-only LLM prompt."""

    ALLOWED_ACTIONS = [
        "analyze_request",
        "ask_clarifying_question",
        "retrieve_evidence",
        "verify_evidence",
        "create_task",
        "create_plan",
        "inspect_code",
        "run_tests",
        "explain_code",
        "propose_code_change",
        "update_documentation",
        "summarize_information",
        "review_security",
        "check_world_briefing",
        "check_cyber_alerts",
        "draft_email",
        "read_calendar",
        "open_website",
        "unknown",
    ]
    BLOCKED_ACTIONS = [
        "execute_shell",
        "delete_all_files",
        "exfiltrate_data",
        "reveal_secret",
        "bypass_security",
        "disable_safety",
        "send_credentials",
        "make_payment",
        "unlock_door",
        "install_malware",
    ]

    def build_messages(
        self,
        raw_input: str,
        context: dict | None = None,
    ) -> list[LLMMessage]:
        """Return system and user messages for safe plan proposal generation."""
        system = (
            "You are only proposing a plan. You do not execute tools and do not "
            "claim actions are completed. Return JSON only. Create safe, concrete "
            "steps, never include credentials, never weaken approval, and never "
            "create destructive steps. External impact, money, sent email, file "
            "deletion, door access, system commands, private data, credentials, "
            "and security changes require approval or high risk. Allowed actions: "
            f"{', '.join(self.ALLOWED_ACTIONS)}. Blocked actions: "
            f"{', '.join(self.BLOCKED_ACTIONS)}. If unsafe or unclear, propose "
            "clarification rather than execution. Use this shape: "
            '{"goal":"...","summary":"...","confidence":0.0,'
            '"reasoning_summary":"...","steps":[{"order":1,"title":"...",'
            '"description":"...","action":"...","target":"...",'
            '"expected_output":"...","risk_level":"low",'
            '"requires_approval":false}]}'
        )
        user = f"Request: {raw_input}"
        if context:
            user += f"\nContext: {json.dumps(context, sort_keys=True, default=str)}"
        return [
            LLMMessage(role=LLMRole.SYSTEM, content=system),
            LLMMessage(role=LLMRole.USER, content=user),
        ]
