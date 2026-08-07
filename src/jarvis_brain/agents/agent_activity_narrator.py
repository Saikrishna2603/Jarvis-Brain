from typing import Any
from uuid import uuid4

from jarvis_platform.schemas.agent_lifecycle import (
    AgentEvent,
    AgentEventType,
    AgentLifecycleStatus,
    AgentRecord,
)
from jarvis_platform.schemas.agent_narration import (
    AgentActivityNarration,
    AgentNarrationKind,
    AgentNarrationTone,
)
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter


class AgentActivityNarrator:
    """Create deterministic, safe UI captions from explicit lifecycle events."""

    serious_context_markers = {
        "medical",
        "legal",
        "emergency",
        "distress",
        "credential",
        "secret",
        "security_critical",
        "high",
        "critical",
        "approval",
        "warning",
        "failure",
    }

    def __init__(
        self,
        safe_logging_filter: SafeLoggingFilter | None = None,
        humor_enabled: bool = False,
        narration_density: str = "normal",
    ) -> None:
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()
        self.humor_enabled = humor_enabled
        self.narration_density = narration_density

    def narrate(
        self,
        event: AgentEvent,
        agent: AgentRecord | None = None,
        context: dict[str, Any] | None = None,
    ) -> AgentActivityNarration:
        """Return one concise caption without exposing private reasoning."""
        safe_context = self.safe_logging_filter.sanitize_metadata(context or {})
        risk_context = str(
            safe_context.get("risk_context")
            or event.metadata.get("risk_context")
            or "low"
        )
        demo = bool(event.metadata.get("demo") or (agent and agent.metadata.get("demo")))
        text, kind, tone = self._caption_for_event(event, agent)

        reason_text = self._explicit_reason(event.metadata, safe_context)
        if reason_text:
            text = f"Why this? {reason_text}"
            kind = AgentNarrationKind.REASON
            tone = AgentNarrationTone.CAUTIOUS

        if self._can_use_humor(event, risk_context) and not reason_text:
            humor_text = self._humor_for_event(event)
            if humor_text:
                text = humor_text
                kind = AgentNarrationKind.HUMOR
                tone = AgentNarrationTone.LIGHT_HUMOR

        safe_text = self.safe_logging_filter.sanitize_message(text)
        return AgentActivityNarration(
            narration_id=f"agent_narration_{uuid4().hex}",
            agent_id=event.agent_id,
            lifecycle_event_id=event.event_id,
            kind=kind,
            tone=tone,
            text=safe_text,
            short_text=safe_text[:96],
            source_event_type=event.event_type.value,
            risk_context=self.safe_logging_filter.sanitize_message(risk_context),
            humor_used=kind == AgentNarrationKind.HUMOR,
            demo=demo,
            metadata={
                "agent_name": agent.name if agent else None,
                "role": agent.role.value if agent else None,
                "narration_density": self.narration_density,
            },
        )

    def _caption_for_event(
        self, event: AgentEvent, agent: AgentRecord | None
    ) -> tuple[str, AgentNarrationKind, AgentNarrationTone]:
        if event.event_type == AgentEventType.AGENT_CREATED:
            return "Coming online...", AgentNarrationKind.STATUS, AgentNarrationTone.NEUTRAL
        if event.event_type == AgentEventType.NAME_ASSIGNED:
            name = agent.name if agent else "agent"
            return f"Identity assigned: {name}.", AgentNarrationKind.STATUS, AgentNarrationTone.FOCUSED
        if event.event_type == AgentEventType.ROLE_ASSIGNED:
            role = agent.role.value if agent else "unknown"
            return f"Role confirmed: {role}.", AgentNarrationKind.STATUS, AgentNarrationTone.FOCUSED
        if event.event_type == AgentEventType.TASK_ASSIGNED:
            return "Reviewing the assignment...", AgentNarrationKind.STATUS, AgentNarrationTone.FOCUSED
        if event.event_type == AgentEventType.TOOL_CONSIDERED:
            return "Checking whether a tool is needed...", AgentNarrationKind.STATUS, AgentNarrationTone.CAUTIOUS
        if event.event_type == AgentEventType.TOOL_CALLED:
            tool_name = self.safe_logging_filter.sanitize_message(
                str(event.metadata.get("tool_name") or "approved tool")
            )
            return f"Using {tool_name}...", AgentNarrationKind.PROGRESS, AgentNarrationTone.FOCUSED
        if event.event_type == AgentEventType.APPROVAL_REQUESTED:
            return "Paused. Your approval is required.", AgentNarrationKind.APPROVAL, AgentNarrationTone.WARNING
        if event.event_type == AgentEventType.APPROVAL_RESOLVED:
            return "Approval decision received.", AgentNarrationKind.APPROVAL, AgentNarrationTone.NEUTRAL
        if event.event_type == AgentEventType.RESULT_CREATED:
            return "Preparing the final result...", AgentNarrationKind.PROGRESS, AgentNarrationTone.CONFIDENT
        if event.event_type == AgentEventType.AGENT_COMPLETED:
            return "Task complete.", AgentNarrationKind.COMPLETION, AgentNarrationTone.CONFIDENT
        if event.event_type == AgentEventType.AGENT_FAILED:
            return "The task failed. Reviewing what went wrong.", AgentNarrationKind.FAILURE, AgentNarrationTone.WARNING
        if event.event_type == AgentEventType.AGENT_TERMINATED:
            return "The agent was stopped.", AgentNarrationKind.FAILURE, AgentNarrationTone.SERIOUS
        if event.event_type == AgentEventType.AGENT_ARCHIVED:
            return "Lifecycle archived.", AgentNarrationKind.COMPLETION, AgentNarrationTone.NEUTRAL
        if event.event_type == AgentEventType.MESSAGE_SENT:
            return "Sharing findings with another specialist...", AgentNarrationKind.HANDOFF, AgentNarrationTone.FOCUSED

        if event.event_type == AgentEventType.STATUS_CHANGED and event.status:
            return self._caption_for_status(event.status)

        return "Lifecycle event received.", AgentNarrationKind.STATUS, AgentNarrationTone.NEUTRAL

    @staticmethod
    def _caption_for_status(
        status: AgentLifecycleStatus,
    ) -> tuple[str, AgentNarrationKind, AgentNarrationTone]:
        mapping = {
            AgentLifecycleStatus.THINKING: ("Considering the safest approach...", AgentNarrationKind.STATUS, AgentNarrationTone.FOCUSED),
            AgentLifecycleStatus.WORKING: ("Working through the current step...", AgentNarrationKind.PROGRESS, AgentNarrationTone.FOCUSED),
            AgentLifecycleStatus.COMMUNICATING: ("Sharing findings with another specialist...", AgentNarrationKind.HANDOFF, AgentNarrationTone.FOCUSED),
            AgentLifecycleStatus.REVIEWING: ("Reviewing the result before returning it...", AgentNarrationKind.PROGRESS, AgentNarrationTone.CAUTIOUS),
            AgentLifecycleStatus.WAITING_FOR_INPUT: ("More information is needed.", AgentNarrationKind.QUESTION, AgentNarrationTone.CAUTIOUS),
            AgentLifecycleStatus.WAITING_FOR_APPROVAL: ("Paused. Your approval is required.", AgentNarrationKind.APPROVAL, AgentNarrationTone.WARNING),
            AgentLifecycleStatus.COMPLETED: ("Task complete.", AgentNarrationKind.COMPLETION, AgentNarrationTone.CONFIDENT),
            AgentLifecycleStatus.FAILED: ("The task failed. Reviewing what went wrong.", AgentNarrationKind.FAILURE, AgentNarrationTone.WARNING),
            AgentLifecycleStatus.TERMINATED: ("The agent was stopped.", AgentNarrationKind.FAILURE, AgentNarrationTone.SERIOUS),
            AgentLifecycleStatus.ARCHIVED: ("Lifecycle archived.", AgentNarrationKind.COMPLETION, AgentNarrationTone.NEUTRAL),
        }
        return mapping.get(status, ("Coming online...", AgentNarrationKind.STATUS, AgentNarrationTone.NEUTRAL))

    def _explicit_reason(
        self, metadata: dict[str, Any], context: dict[str, Any]
    ) -> str | None:
        for key in (
            "reason_summary",
            "validation_reason",
            "approval_reason",
            "source_trust_reason",
            "plan_step_reason",
        ):
            value = metadata.get(key) or context.get(key)
            if isinstance(value, str) and value.strip():
                return self.safe_logging_filter.sanitize_message(value.strip())[:210]
        return None

    def _can_use_humor(self, event: AgentEvent, risk_context: str) -> bool:
        if not self.humor_enabled:
            return False
        if event.event_type in {
            AgentEventType.APPROVAL_REQUESTED,
            AgentEventType.AGENT_FAILED,
            AgentEventType.AGENT_TERMINATED,
        }:
            return False
        context = f"{risk_context} {event.message}".lower()
        return not any(marker in context for marker in self.serious_context_markers)

    @staticmethod
    def _humor_for_event(event: AgentEvent) -> str | None:
        if event.event_type == AgentEventType.AGENT_COMPLETED:
            return "Task complete. No dramatic explosions detected."
        if event.event_type == AgentEventType.STATUS_CHANGED and event.status == AgentLifecycleStatus.REVIEWING:
            return "Rechecking the result. Arrogance remains disabled."
        if event.event_type == AgentEventType.STATUS_CHANGED and event.status == AgentLifecycleStatus.WORKING:
            return "Working through the step. Optimism remains cautiously enabled."
        return None
