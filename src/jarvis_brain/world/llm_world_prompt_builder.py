import json
from typing import Any

from jarvis_platform.schemas.llm import LLMMessage, LLMRole


class LLMWorldPromptBuilder:
    """Build a constrained prompt for world-intelligence briefing refinement."""

    def build_messages(
        self,
        briefing_type: str,
        events: list | None = None,
        suggestions: list | None = None,
        alerts: list | None = None,
        context: dict | None = None,
    ) -> list[LLMMessage]:
        """Return system and user messages for safe world briefing refinement."""
        system = (
            "You are refining a world intelligence briefing only. You do not "
            "fetch live data. You do not invent events, counts, sources, or "
            "citations. Use only the provided events, suggestions, and alerts. "
            "If data is mock or placeholder, clearly say it is mock or placeholder. "
            "Do not exaggerate certainty. Do not reveal secrets. Do not create "
            "executable actions. Suggested next steps must be non-executing "
            "recommendations only. Anything requiring real action must say approval "
            "is required. Return JSON only. Use this shape: "
            '{"summary":"...","priority_items":[],"alerts":[],'
            '"project_relevance":[],"suggested_next_steps":[],'
            '"evidence_event_ids":[],"confidence":0.0,"risk_flags":[]}'
        )
        payload = {
            "briefing_type": briefing_type,
            "context": context or {},
            "events": [self._dump_item(event) for event in events or []],
            "suggestions": [
                self._dump_item(suggestion) for suggestion in suggestions or []
            ],
            "alerts": [self._dump_item(alert) for alert in alerts or []],
        }
        return [
            LLMMessage(role=LLMRole.SYSTEM, content=system),
            LLMMessage(
                role=LLMRole.USER,
                content=json.dumps(payload, sort_keys=True, default=str),
            ),
        ]

    def _dump_item(self, item: Any) -> dict[str, Any]:
        """Serialize Pydantic or dictionary items for prompt context."""
        if hasattr(item, "model_dump"):
            return item.model_dump(mode="json")
        if isinstance(item, dict):
            return item
        return {"value": str(item)}
