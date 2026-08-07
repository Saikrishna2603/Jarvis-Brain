from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory, WorldEventSeverity
from jarvis_platform.schemas.world_impact import WorldImpactAnalysis
from jarvis_brain.world.relevance_scorer import RelevanceScorer


class ImpactAnalysisEngine:
    """Create rule-based impact analysis for Jarvis world events."""

    SECURITY_INTEGRATION_TAGS = {"cloud", "iam", "api", "secrets"}

    def __init__(self, relevance_scorer: RelevanceScorer | None = None) -> None:
        """Create the engine with a rule-based relevance scorer."""
        self.relevance_scorer = relevance_scorer or RelevanceScorer()

    def analyze_event(
        self,
        event: WorldEvent,
        context: dict | None = None,
    ) -> WorldImpactAnalysis:
        """Analyze one world event and return a structured impact analysis."""
        relevance_score = self._relevance_score(event=event, context=context)
        tags = {tag.lower() for tag in event.tags}

        affected_areas: list[str] = []
        possible_outcomes: list[str] = []
        suggested_actions: list[str] = []
        impact_summary = "Impact is unclear or low relevance for the current project."
        risk_level = "low"

        if event.category == WorldEventCategory.CYBERSECURITY:
            impact_summary = "This may affect Jarvis security posture and integration safety."
            affected_areas = ["security"]
            possible_outcomes = [
                "Increased risk from misconfiguration or exposed credentials.",
                "Need for stronger permission and integration controls.",
            ]
            suggested_actions = [
                "Review SecretGuard and SourceTrustManager requirements.",
                "Review permissions and OAuth handling for related integrations.",
            ]
            if tags & self.SECURITY_INTEGRATION_TAGS:
                affected_areas.extend(["cloud", "Jarvis integrations"])
            risk_level = self._risk_from_severity(event.severity, medium_for_medium=True)

        elif event.category == WorldEventCategory.AI_RESEARCH:
            impact_summary = "This may inform future Jarvis architecture or development workflow."
            affected_areas = [
                "Jarvis project",
                "agent runtime",
                "planner",
                "developer productivity",
            ]
            possible_outcomes = [
                "Possible improvements to Jarvis architecture or agent workflows.",
                "Potential productivity gains for future development phases.",
            ]
            suggested_actions = ["Review the framework or research later."]
            risk_level = "medium" if event.severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL} else "low"

        elif event.category in {WorldEventCategory.MARKETS, WorldEventCategory.FINANCE}:
            impact_summary = "This may affect finance, budgeting, or market awareness."
            affected_areas = ["finance", "budgeting", "market awareness"]
            possible_outcomes = [
                "Volatility or uncertainty may affect planning.",
                "Market movement may be worth monitoring over time.",
            ]
            suggested_actions = [
                "Create a briefing or monitor the trend without treating this as financial advice."
            ]
            risk_level = "medium" if event.severity in {WorldEventSeverity.MEDIUM, WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL} else "low"

        elif event.category in {WorldEventCategory.WEATHER, WorldEventCategory.AVIATION}:
            impact_summary = "This may affect travel, schedules, or local planning."
            affected_areas = ["travel", "schedule", "local planning"]
            possible_outcomes = [
                "Possible delays or disruptions.",
                "Plans may need to be checked if travel is involved.",
            ]
            if self._travel_is_relevant(event=event, context=context):
                suggested_actions = ["Check calendar or travel plans for possible conflicts."]
            risk_level = self._risk_from_severity(event.severity, medium_for_medium=True)

        elif event.category in {
            WorldEventCategory.GEOPOLITICS,
            WorldEventCategory.ENERGY,
            WorldEventCategory.SUPPLY_CHAIN,
        }:
            impact_summary = "This may affect global risk, markets, or supply chain awareness."
            affected_areas = ["markets", "supply chain", "global risk"]
            possible_outcomes = [
                "Uncertainty or downstream effects may increase.",
                "Related markets or supply chains may need monitoring.",
            ]
            suggested_actions = ["Monitor updates before taking immediate action."]
            risk_level = self._risk_from_severity(event.severity, medium_for_medium=True)

        requires_user_discussion = self._requires_user_discussion(
            event=event,
            relevance_score=relevance_score,
        )
        should_create_task = self._should_create_task(
            event=event,
            context=context,
            requires_user_discussion=requires_user_discussion,
        )

        return WorldImpactAnalysis(
            analysis_id=f"impact-{event.event_id}",
            event_id=event.event_id,
            event_title=event.title,
            category=event.category.value,
            severity=event.severity.value,
            relevance_score=relevance_score,
            impact_summary=impact_summary,
            affected_areas=affected_areas,
            possible_outcomes=possible_outcomes,
            suggested_actions=suggested_actions,
            risk_level=risk_level,
            confidence_score=event.confidence_score,
            requires_user_discussion=requires_user_discussion,
            should_create_task=should_create_task,
            metadata={
                "source_name": event.source_name,
                "tags": event.tags,
            },
        )

    def analyze_events(
        self,
        events: list[WorldEvent],
        context: dict | None = None,
    ) -> list[WorldImpactAnalysis]:
        """Analyze several world events."""
        return [self.analyze_event(event=event, context=context) for event in events]

    def summarize_analysis(self, analysis: WorldImpactAnalysis) -> str:
        """Return a short natural-language summary of one analysis."""
        summary = f"{analysis.event_title}: {analysis.impact_summary}"
        if analysis.suggested_actions:
            summary = f"{summary} Suggested action: {analysis.suggested_actions[0]}"

        return summary

    def _relevance_score(self, event: WorldEvent, context: dict | None) -> float:
        """Use the event score when present, otherwise score it now."""
        if event.relevance_score > 0:
            return event.relevance_score

        return self.relevance_scorer.score_event(event=event, context=context)

    def _requires_user_discussion(
        self,
        event: WorldEvent,
        relevance_score: float,
    ) -> bool:
        """Return True when the event should be discussed with the user."""
        if relevance_score >= 0.8:
            return True

        if event.should_alert:
            return True

        return (
            event.severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL}
            and event.confidence_score >= 0.5
        )

    def _should_create_task(
        self,
        event: WorldEvent,
        context: dict | None,
        requires_user_discussion: bool,
    ) -> bool:
        """Return True when Jarvis should create a follow-up task later."""
        if context and context.get("create_tasks_for_relevant_world_events") is True:
            return requires_user_discussion

        return requires_user_discussion and event.category in {
            WorldEventCategory.CYBERSECURITY,
            WorldEventCategory.AI_RESEARCH,
        }

    def _travel_is_relevant(self, event: WorldEvent, context: dict | None) -> bool:
        """Return True when weather or aviation should trigger plan checking."""
        interests = {str(item).lower() for item in (context or {}).get("interests", [])}
        return (
            {"travel", "weather"} & interests
            or event.severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL}
        )

    def _risk_from_severity(
        self,
        severity: WorldEventSeverity,
        medium_for_medium: bool = False,
    ) -> str:
        """Map world event severity to an impact risk level."""
        if severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL}:
            return "high"

        if medium_for_medium and severity == WorldEventSeverity.MEDIUM:
            return "medium"

        return "low"
