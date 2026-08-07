import json
from uuid import uuid4

from pydantic import ValidationError

from jarvis_platform.schemas.llm_plan import (
    LLMPlanCandidate,
    LLMPlanDecision,
    LLMPlanStepCandidate,
    LLMPlanValidationResult,
)
from jarvis_platform.schemas.plan import ExecutionPlan, PlanStep


class LLMPlanValidator:
    """Parse and validate untrusted LLM plan proposals."""

    ALLOWED_ACTIONS = {
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
    }
    BLOCKED_ACTIONS = {
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
    }
    SUSPICIOUS_PHRASES = {
        "reveal secret",
        "bypass security",
        "disable safety",
        "delete everything",
        "exfiltrate",
        "steal",
        "malware",
        "send credentials",
    }

    def parse_candidate(
        self,
        raw_input: str,
        llm_text: str,
    ) -> LLMPlanCandidate | None:
        """Parse the first valid JSON object without executing its contents."""
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
            steps = data.get("steps")
            if not isinstance(steps, list):
                return None
            normalized_steps = []
            for step in steps:
                if not isinstance(step, dict):
                    return None
                normalized = dict(step)
                normalized.setdefault("step_id", str(uuid4()))
                normalized_steps.append(normalized)
            data["steps"] = normalized_steps
            try:
                return LLMPlanCandidate(
                    candidate_id=str(uuid4()),
                    raw_input=raw_input,
                    **data,
                )
            except (TypeError, ValidationError):
                return None
        return None

    def validate(self, candidate: LLMPlanCandidate) -> LLMPlanValidationResult:
        """Reject unsafe plans and accept only complete allowlisted proposals."""
        if candidate.confidence < 0.55:
            return self._rejected(candidate, "Plan confidence is below 0.55.")
        if not candidate.steps:
            return self._rejected(candidate, "Plan has no steps.")

        rejected_steps = [
            step for step in candidate.steps if not self._step_is_safe(step)
        ]
        if rejected_steps:
            return self._rejected(
                candidate,
                "Plan contains a blocked, unsupported, or suspicious step.",
                rejected_steps,
            )

        metadata = {"low_confidence": True} if candidate.confidence < 0.7 else {}
        return LLMPlanValidationResult(
            decision=LLMPlanDecision.ACCEPTED,
            candidate=candidate,
            reason="All plan steps passed the planning allowlist.",
            accepted_steps=candidate.steps,
            metadata=metadata,
        )

    def to_execution_plan(self, candidate: LLMPlanCandidate) -> ExecutionPlan:
        """Convert a validated proposal into the existing plan contract."""
        steps = [
            PlanStep(
                step_id=step.step_id,
                action=step.action or "unknown",
                target=step.target,
                payload={
                    "title": step.title,
                    "description": step.description,
                    "expected_output": step.expected_output,
                    "proposed_risk_level": step.risk_level.value,
                    **step.metadata,
                },
                requires_approval=step.requires_approval,
                reason=step.description,
            )
            for step in sorted(candidate.steps, key=lambda item: item.order)
        ]
        return ExecutionPlan(
            plan_id=f"llm-plan-{candidate.candidate_id}",
            user_goal=candidate.goal,
            steps=steps,
            status="pending",
            metadata={
                "llm_assisted": True,
                "candidate_id": candidate.candidate_id,
                "summary": candidate.summary,
                "confidence": candidate.confidence,
                "reasoning_summary": candidate.reasoning_summary,
            },
        )

    def _step_is_safe(self, step: LLMPlanStepCandidate) -> bool:
        """Return True only for allowlisted, non-suspicious steps."""
        action = (step.action or "unknown").strip().lower()
        if action in self.BLOCKED_ACTIONS or action not in self.ALLOWED_ACTIONS:
            return False
        text = " ".join(
            value
            for value in (step.title, step.description, step.target or "")
            if value
        ).lower()
        return not any(phrase in text for phrase in self.SUSPICIOUS_PHRASES)

    def _rejected(
        self,
        candidate: LLMPlanCandidate,
        reason: str,
        rejected_steps: list[LLMPlanStepCandidate] | None = None,
    ) -> LLMPlanValidationResult:
        """Build a rejected plan result."""
        return LLMPlanValidationResult(
            decision=LLMPlanDecision.REJECTED,
            candidate=candidate,
            reason=reason,
            rejected_steps=rejected_steps or [],
        )
