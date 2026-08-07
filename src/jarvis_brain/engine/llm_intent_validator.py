import json
from uuid import uuid4

from pydantic import ValidationError

from jarvis_platform.schemas.llm_intent import (
    LLMIntentCandidate,
    LLMIntentDecision,
    LLMIntentValidationResult,
)


class LLMIntentValidator:
    """Validate LLM-proposed intents against a strict allowlist."""

    ALLOWED_ACTIONS: dict[str, set[str]] = {
        "open_website": {"open_website", "open_url"},
        "browser": {"open_website", "search_web"},
        "email": {"draft_email", "send_email", "read_email"},
        "calendar": {"create_event", "read_calendar", "check_availability"},
        "smart_home": {"turn_on", "turn_off", "set_temperature", "check_status"},
        "finance": {"summarize_spending", "check_balance", "review_budget"},
        "task": {"create_task", "list_tasks", "update_task"},
        "plan": {"create_plan", "explain_plan", "continue_plan"},
        "approval_response": {"approve", "reject"},
        "world_intelligence": {
            "get_world_briefing",
            "get_cyber_alerts",
            "get_project_relevant_updates",
            "get_ai_research_updates",
            "get_world_alerts",
        },
        "universal_knowledge": {
            "answer_with_knowledge_flow",
            "ask_clarifying_questions",
            "provide_guidance",
        },
        "coding_help": {"debug_code", "explain_code", "write_code", "review_repo"},
        "general_question": {"answer_general"},
        "unknown": {"unknown"},
    }
    SUSPICIOUS_VALUES = {
        "delete_all",
        "exfiltrate",
        "reveal_secret",
        "bypass_security",
        "disable_safety",
        "execute_shell",
    }

    def validate(self, candidate: LLMIntentCandidate) -> LLMIntentValidationResult:
        """Return a safe validation decision for one candidate."""
        intent_type = candidate.intent_type.strip().lower()
        action = candidate.action.strip().lower() if candidate.action else None
        target = candidate.target.strip() if candidate.target else None

        if self._contains_suspicious_value(action, target):
            return self._rejected(candidate, "Candidate contains a prohibited value.")
        if intent_type not in self.ALLOWED_ACTIONS:
            return self._rejected(candidate, "Intent type is not allowed.")
        if intent_type == "unknown":
            return LLMIntentValidationResult(
                decision=LLMIntentDecision.FALLBACK_TO_UNKNOWN,
                candidate=candidate,
                reason="The classifier could not identify a supported intent.",
                sanitized_intent_type="unknown",
                sanitized_action="unknown",
            )
        if action not in self.ALLOWED_ACTIONS[intent_type]:
            return self._rejected(candidate, "Action is not allowed for this intent type.")
        if candidate.confidence < 0.55:
            return self._rejected(candidate, "Candidate confidence is below 0.55.")

        metadata = {"low_confidence": True} if candidate.confidence < 0.7 else {}
        return LLMIntentValidationResult(
            decision=LLMIntentDecision.ACCEPTED,
            candidate=candidate,
            reason="Candidate passed the intent allowlist.",
            sanitized_intent_type=intent_type,
            sanitized_action=action,
            sanitized_target=target,
            metadata=metadata,
        )

    def parse_candidate(
        self,
        raw_input: str,
        llm_text: str,
    ) -> LLMIntentCandidate | None:
        """Parse the first valid JSON object from an LLM response."""
        decoder = json.JSONDecoder()
        for index, character in enumerate(llm_text):
            if character != "{":
                continue
            try:
                data, _ = decoder.raw_decode(llm_text[index:])
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            try:
                return LLMIntentCandidate(
                    candidate_id=str(uuid4()),
                    raw_input=raw_input,
                    **data,
                )
            except (TypeError, ValidationError):
                return None
        return None

    def _contains_suspicious_value(
        self,
        action: str | None,
        target: str | None,
    ) -> bool:
        """Return True if action or target contains a prohibited phrase."""
        values = (action or "", (target or "").lower().replace(" ", "_"))
        return any(
            prohibited in value
            for value in values
            for prohibited in self.SUSPICIOUS_VALUES
        )

    def _rejected(
        self,
        candidate: LLMIntentCandidate,
        reason: str,
    ) -> LLMIntentValidationResult:
        """Build a rejected validation result."""
        return LLMIntentValidationResult(
            decision=LLMIntentDecision.REJECTED,
            candidate=candidate,
            reason=reason,
        )
