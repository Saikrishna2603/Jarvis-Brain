from jarvis_platform.schemas.world_impact import WorldImpactAnalysis
from jarvis_platform.schemas.world_suggestion import (
    SuggestionPriority,
    SuggestionType,
    WorldSuggestion,
)


class SuggestionEngine:
    """Create rule-based recommendations from world impact analyses.

    Suggestions are advisory only. This engine does not execute tools, create
    tasks, call APIs, or use LLMs.
    """

    def create_suggestion_from_analysis(
        self,
        analysis: WorldImpactAnalysis,
    ) -> WorldSuggestion:
        """Create one recommendation from an impact analysis."""
        if self._should_escalate(analysis):
            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.ESCALATE_ALERT,
                priority=SuggestionPriority.URGENT,
                message="This should be brought to the user's attention.",
                rationale="Very high relevance and user discussion required.",
            )

        if self.should_ignore_analysis(analysis):
            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.IGNORE,
                priority=SuggestionPriority.LOW,
                message="No action is needed right now.",
                rationale="Low relevance and no follow-up flags.",
            )

        if analysis.category == "cybersecurity":
            if analysis.should_create_task:
                return self._build_suggestion(
                    analysis=analysis,
                    suggestion_type=SuggestionType.CREATE_TASK,
                    priority=SuggestionPriority.HIGH,
                    message="Create a security review task for this event.",
                    suggested_action="create_security_review_task",
                    rationale="Cybersecurity impact analysis recommends follow-up.",
                )

            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.REVIEW_SECURITY,
                priority=SuggestionPriority.HIGH,
                message="Review security controls or consider creating a security task.",
                rationale="Cybersecurity event may affect Jarvis security posture.",
            )

        if analysis.category == "ai_research":
            priority = SuggestionPriority.HIGH
            if analysis.relevance_score < 0.85:
                priority = SuggestionPriority.MEDIUM

            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.REVIEW_RESEARCH,
                priority=priority,
                message="Review this AI update for possible Jarvis improvements.",
                rationale="AI research may inform future Jarvis architecture.",
            )

        if analysis.category in {"markets", "finance"}:
            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.REVIEW_FINANCE,
                priority=SuggestionPriority.MEDIUM,
                message="Monitor this event or add it to a finance briefing.",
                rationale="Finance and market events may affect planning awareness.",
            )

        if analysis.category in {"weather", "aviation"}:
            priority = SuggestionPriority.HIGH
            if analysis.risk_level != "high" and analysis.relevance_score < 0.8:
                priority = SuggestionPriority.MEDIUM

            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=SuggestionType.REVIEW_TRAVEL,
                priority=priority,
                message="Check calendar or travel plans if this is relevant.",
                rationale="Weather or aviation impact may affect schedule planning.",
            )

        if analysis.category in {"geopolitics", "energy", "supply_chain"}:
            suggestion_type = SuggestionType.MONITOR
            priority = SuggestionPriority.LOW
            message = "Monitor updates for downstream effects."
            if analysis.relevance_score >= 0.5:
                suggestion_type = SuggestionType.ADD_TO_BRIEFING
                priority = SuggestionPriority.MEDIUM
                message = "Add this to a briefing and monitor updates."

            return self._build_suggestion(
                analysis=analysis,
                suggestion_type=suggestion_type,
                priority=priority,
                message=message,
                rationale="Global risk events may affect markets or supply chains.",
            )

        return self._build_suggestion(
            analysis=analysis,
            suggestion_type=SuggestionType.IGNORE,
            priority=SuggestionPriority.LOW,
            message="No action is needed right now.",
            rationale="Impact is low or unclear.",
        )

    def create_suggestions_from_analyses(
        self,
        analyses: list[WorldImpactAnalysis],
    ) -> list[WorldSuggestion]:
        """Create recommendations for several analyses."""
        return [self.create_suggestion_from_analysis(analysis) for analysis in analyses]

    def should_ignore_analysis(self, analysis: WorldImpactAnalysis) -> bool:
        """Return True when an analysis does not need follow-up."""
        return (
            analysis.relevance_score < 0.25
            and not analysis.requires_user_discussion
            and not analysis.should_create_task
        )

    def summarize_suggestion(self, suggestion: WorldSuggestion) -> str:
        """Return a short natural-language suggestion summary."""
        return f"{suggestion.title} [{suggestion.priority.value}]: {suggestion.message}"

    def _should_escalate(self, analysis: WorldImpactAnalysis) -> bool:
        """Return True when a suggestion should become an urgent alert."""
        return analysis.requires_user_discussion and analysis.relevance_score >= 0.9

    def _build_suggestion(
        self,
        analysis: WorldImpactAnalysis,
        suggestion_type: SuggestionType,
        priority: SuggestionPriority,
        message: str,
        suggested_action: str | None = None,
        rationale: str | None = None,
    ) -> WorldSuggestion:
        """Build a deterministic world suggestion from analysis details."""
        return WorldSuggestion(
            suggestion_id=f"suggestion-{analysis.analysis_id}",
            event_id=analysis.event_id,
            analysis_id=analysis.analysis_id,
            title=f"Suggestion for {analysis.event_title}",
            message=message,
            suggestion_type=suggestion_type,
            priority=priority,
            requires_user_approval=False,
            suggested_action=suggested_action,
            target=analysis.event_id,
            rationale=rationale,
            metadata={
                "category": analysis.category,
                "risk_level": analysis.risk_level,
                "relevance_score": analysis.relevance_score,
            },
        )
