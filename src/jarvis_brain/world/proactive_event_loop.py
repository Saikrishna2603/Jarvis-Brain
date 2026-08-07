from jarvis_brain.ports import WorldEventMemoryManager
from jarvis_brain.ports import WorldEventRepository
from jarvis_platform.schemas.world_event import WorldEvent
from jarvis_platform.schemas.world_suggestion import (
    SuggestionPriority,
    SuggestionType,
    WorldSuggestion,
)
from jarvis_brain.ports import WorldDataDriver
from jarvis_brain.world.impact_analysis_engine import ImpactAnalysisEngine
from jarvis_brain.world.relevance_scorer import RelevanceScorer
from jarvis_brain.world.suggestion_engine import SuggestionEngine
from jarvis_brain.world.world_intelligence_repository import WorldIntelligenceRepository


class ProactiveEventLoop:
    """One-shot mock world intelligence loop for Jarvis v1.

    This loop does not schedule background work or call real APIs. It collects
    mock world events from WorldDataDriver, scores them, analyzes impact, and
    creates advisory suggestions.
    """

    def __init__(
        self,
        world_event_repository: WorldEventRepository | None = None,
        world_intelligence_repository: WorldIntelligenceRepository | None = None,
    ) -> None:
        """Create the mock world intelligence components."""
        self.world_data_driver = WorldDataDriver()
        self.event_memory_manager = WorldEventMemoryManager()
        self.relevance_scorer = RelevanceScorer()
        self.impact_analysis_engine = ImpactAnalysisEngine(self.relevance_scorer)
        self.suggestion_engine = SuggestionEngine()
        self.latest_suggestions: list[WorldSuggestion] = []
        self.world_event_repository = world_event_repository
        self.world_intelligence_repository = world_intelligence_repository

    def run_once(self, context: dict | None = None) -> dict:
        """Run one mock world intelligence collection and analysis cycle."""
        raw_events = self.world_data_driver.get_global_briefing()
        scored_events = self.relevance_scorer.score_events(raw_events, context=context)
        self.event_memory_manager.save_events(scored_events)
        if self.world_event_repository is not None:
            self.world_event_repository.save_events(scored_events)

        analyses = self.impact_analysis_engine.analyze_events(
            scored_events,
            context=context,
        )
        self.latest_suggestions = self.suggestion_engine.create_suggestions_from_analyses(
            analyses
        )
        if self.world_intelligence_repository is not None:
            self.world_intelligence_repository.save_analyses(analyses)
            self.world_intelligence_repository.save_suggestions(self.latest_suggestions)

        return {
            "status": "success",
            "events_collected": len(scored_events),
            "events_stored": len(self.event_memory_manager.get_all_events()),
            "analyses_created": len(analyses),
            "suggestions_created": len(self.latest_suggestions),
            "alerts_created": len(self.get_alert_suggestions()),
            "events": self._serialize_events(scored_events),
            "suggestions": self._serialize_suggestions(self.latest_suggestions),
        }

    def get_daily_briefing(self) -> dict:
        """Return a simple briefing from stored events and latest suggestions."""
        events = self.event_memory_manager.get_all_events()
        high_priority_events = [
            event for event in events if event.is_high_priority()
        ]
        alert_suggestions = self.get_alert_suggestions()

        return {
            "title": "Jarvis World Intelligence Briefing",
            "events_count": len(events),
            "high_priority_count": len(high_priority_events),
            "suggestions_count": len(self.latest_suggestions),
            "alerts_count": len(alert_suggestions),
            "high_priority_events": self._serialize_events(high_priority_events),
            "alert_suggestions": self._serialize_suggestions(alert_suggestions),
        }

    def get_stored_events(self) -> list[WorldEvent]:
        """Return all world events stored by the loop."""
        return self.event_memory_manager.get_all_events()

    def get_alert_suggestions(self) -> list[WorldSuggestion]:
        """Return suggestions that should be surfaced as alerts."""
        return [
            suggestion
            for suggestion in self.latest_suggestions
            if suggestion.suggestion_type == SuggestionType.ESCALATE_ALERT
            or suggestion.priority == SuggestionPriority.URGENT
            or suggestion.needs_user_response()
        ]

    def _serialize_events(self, events: list[WorldEvent]) -> list[dict]:
        """Serialize world events for API-friendly responses."""
        return [event.model_dump(mode="json") for event in events]

    def _serialize_suggestions(self, suggestions: list[WorldSuggestion]) -> list[dict]:
        """Serialize world suggestions for API-friendly responses."""
        return [suggestion.model_dump(mode="json") for suggestion in suggestions]
