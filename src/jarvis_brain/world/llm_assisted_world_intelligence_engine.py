import os
from typing import Any

from jarvis_platform.config import load_app_environment
from jarvis_brain.llm.safe_llm_service import SafeLLMService
from jarvis_brain.world.llm_world_prompt_builder import LLMWorldPromptBuilder
from jarvis_brain.world.llm_world_validator import LLMWorldValidator


class LLMAssistedWorldIntelligenceEngine:
    """Refine rule-based world briefings without fetching data or executing tools."""

    def __init__(
        self,
        safe_llm_service: SafeLLMService | None = None,
        prompt_builder: LLMWorldPromptBuilder | None = None,
        validator: LLMWorldValidator | None = None,
        enabled: bool = False,
        llm_confidence_threshold: float = 0.55,
    ) -> None:
        """Create a disabled-by-default world briefing refinement engine."""
        self.safe_llm_service = safe_llm_service or SafeLLMService()
        self.prompt_builder = prompt_builder or LLMWorldPromptBuilder()
        self.validator = validator or LLMWorldValidator()
        self.enabled = enabled
        self.llm_confidence_threshold = llm_confidence_threshold

    def create_briefing(
        self,
        briefing_type: str = "world_briefing",
        events: list | None = None,
        suggestions: list | None = None,
        alerts: list | None = None,
        context: dict | None = None,
        base_summary: str | None = None,
    ) -> dict[str, Any]:
        """Return a base or safely refined world briefing dictionary."""
        base = self._base_briefing(
            briefing_type,
            base_summary,
            events,
            suggestions,
            alerts,
        )
        if not self.enabled:
            return base

        response = self.safe_llm_service.generate(
            self.prompt_builder.build_messages(
                briefing_type=briefing_type,
                events=events,
                suggestions=suggestions,
                alerts=alerts,
                context=context,
            ),
            metadata={"task_type": "world_intelligence"},
        )
        if not response.is_success():
            return self._with_rejection(base, "LLM world intelligence was unavailable.", [])

        candidate = self.validator.parse_candidate(briefing_type, response.content)
        if candidate is None:
            return self._with_rejection(base, "LLM world briefing was not valid JSON.", [])
        if candidate.confidence < self.llm_confidence_threshold:
            return self._with_rejection(
                base,
                "LLM world briefing confidence was below the configured threshold.",
                [],
            )

        validation = self.validator.validate(candidate, events, suggestions, alerts)
        if not validation.is_accepted():
            return self._with_rejection(
                base,
                validation.reason,
                [flag.value for flag in validation.risk_flags],
            )

        return {
            "summary": validation.sanitized_summary or candidate.summary,
            "priority_items": candidate.priority_items,
            "alerts": candidate.alerts,
            "project_relevance": candidate.project_relevance,
            "suggested_next_steps": candidate.suggested_next_steps,
            "evidence_event_ids": candidate.evidence_event_ids,
            "metadata": {
                **base["metadata"],
                "llm_assisted": True,
                "llm_assisted_world": True,
                "llm_model": response.model,
                "world_model": response.model,
                "llm_confidence": candidate.confidence,
                "briefing_type": briefing_type,
                "tools_executed": False,
                "live_data_fetched": False,
            },
        }

    def _base_briefing(
        self,
        briefing_type: str,
        base_summary: str | None,
        events: list | None,
        suggestions: list | None,
        alerts: list | None,
    ) -> dict[str, Any]:
        return {
            "summary": base_summary or "World intelligence briefing is available.",
            "priority_items": [],
            "alerts": [],
            "project_relevance": [],
            "suggested_next_steps": [],
            "evidence_event_ids": [],
            "metadata": {
                "llm_assisted": False,
                "briefing_type": briefing_type,
                "events_count": len(events or []),
                "suggestions_count": len(suggestions or []),
                "alerts_count": len(alerts or []),
            },
        }

    def _with_rejection(
        self,
        base: dict[str, Any],
        reason: str,
        risk_flags: list[str],
    ) -> dict[str, Any]:
        metadata = dict(base.get("metadata", {}))
        metadata.update(
            {
                "llm_assisted": False,
                "llm_world_rejected": True,
                "llm_reject_reason": reason,
                "risk_flags": risk_flags,
            }
        )
        return {**base, "metadata": metadata}


def create_llm_assisted_world_intelligence_engine(
    safe_llm_service: SafeLLMService | None = None,
) -> LLMAssistedWorldIntelligenceEngine:
    """Create an assisted world engine from environment configuration."""
    load_app_environment()
    return LLMAssistedWorldIntelligenceEngine(
        safe_llm_service=safe_llm_service,
        enabled=_env_truthy(os.getenv("LLM_WORLD_ENABLED")),
        llm_confidence_threshold=_float_from_env(
            "LLM_WORLD_CONFIDENCE_THRESHOLD",
            0.55,
        ),
    )


def _env_truthy(value: str | None) -> bool:
    """Return True for common truthy environment values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    """Read a bounded confidence threshold from the environment."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(1.0, max(0.0, value))
