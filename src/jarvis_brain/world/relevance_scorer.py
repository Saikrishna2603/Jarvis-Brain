from typing import Any

from jarvis_platform.schemas.world_event import (
    VerificationStatus,
    WorldEvent,
    WorldEventCategory,
    WorldEventSeverity,
)


class RelevanceScorer:
    """Rule-based relevance scorer for Jarvis world events.

    The scorer estimates how relevant a public/global event is to the user's
    current project and interests. It does not perform prediction, impact
    analysis, or LLM reasoning.
    """

    DEFAULT_INTERESTS = {
        "ai",
        "ai agents",
        "agents",
        "cybersecurity",
        "cloud security",
        "cloud",
        "iam",
        "jarvis project",
        "jarvis",
        "software development",
        "software",
        "development",
        "finance",
    }

    HIGH_RELEVANCE_SECURITY_TAGS = {
        "cloud",
        "iam",
        "vulnerability",
        "breach",
        "malware",
        "phishing",
    }
    HIGH_RELEVANCE_AI_TAGS = {"ai", "agents", "frameworks", "llm"}
    HIGH_RELEVANCE_TECH_TAGS = {"software", "development", "cloud", "api"}
    MARKET_IMPACT_TAGS = {"markets", "market", "finance", "supply_chain", "supply chain"}

    def score_event(self, event: WorldEvent, context: dict | None = None) -> float:
        """Return a relevance score from 0.0 to 1.0 for one event."""
        explanation = self._score_with_reasons(event=event, context=context)
        return explanation["relevance_score"]

    def explain_relevance(
        self,
        event: WorldEvent,
        context: dict | None = None,
    ) -> dict[str, Any]:
        """Explain why an event received its relevance score."""
        explanation = self._score_with_reasons(event=event, context=context)
        return {
            "event_id": event.event_id,
            "relevance_score": explanation["relevance_score"],
            "reasons": explanation["reasons"],
            "should_alert": explanation["should_alert"],
        }

    def score_events(
        self,
        events: list[WorldEvent],
        context: dict | None = None,
    ) -> list[WorldEvent]:
        """Return copied events with relevance_score and should_alert updated."""
        updated_events: list[WorldEvent] = []
        for event in events:
            explanation = self._score_with_reasons(event=event, context=context)
            updated_events.append(
                event.model_copy(
                    update={
                        "relevance_score": explanation["relevance_score"],
                        "should_alert": explanation["should_alert"],
                    }
                )
            )

        return updated_events

    def _score_with_reasons(
        self,
        event: WorldEvent,
        context: dict | None,
    ) -> dict[str, Any]:
        """Compute score, reasons, and alert status for one event."""
        reasons: list[str] = []
        score = 0.2

        tags = {tag.lower() for tag in event.tags}
        text = self._event_text(event)

        if event.category == WorldEventCategory.CYBERSECURITY:
            score = 0.65
            reasons.append("Cybersecurity events are relevant to Jarvis safety.")
            if tags & self.HIGH_RELEVANCE_SECURITY_TAGS:
                score = 0.9
                reasons.append("Security tags match cloud/IAM/threat interests.")

        elif event.category == WorldEventCategory.AI_RESEARCH:
            score = 0.65
            reasons.append("AI research events are relevant to the Jarvis project.")
            if tags & self.HIGH_RELEVANCE_AI_TAGS:
                score = 0.82
                reasons.append("AI research tags match agents/frameworks interests.")

        elif event.category == WorldEventCategory.TECHNOLOGY:
            score = 0.5
            reasons.append("Technology events may affect software development.")
            if tags & self.HIGH_RELEVANCE_TECH_TAGS:
                score = 0.78
                reasons.append("Technology tags match software/cloud/API interests.")

        elif event.category in {WorldEventCategory.FINANCE, WorldEventCategory.MARKETS}:
            score = 0.55
            reasons.append("Finance and market events are medium relevance.")

        elif event.category in {WorldEventCategory.WEATHER, WorldEventCategory.AVIATION}:
            score = self._score_weather_or_travel(event=event, context=context, reasons=reasons)

        elif event.category in {WorldEventCategory.GEOPOLITICS, WorldEventCategory.ENERGY}:
            score = 0.35
            reasons.append("Geopolitical and energy events are low-to-medium relevance by default.")
            if tags & self.MARKET_IMPACT_TAGS or "markets" in text or "supply chain" in text:
                score = 0.55
                reasons.append("Event may affect markets or supply chain.")

        elif event.category == WorldEventCategory.NEWS:
            score = 0.25
            reasons.append("General news is low relevance by default.")

        if event.severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL} and event.should_alert:
            score = max(score, 0.9)
            reasons.append("High or critical alert event is highly relevant.")

        if event.severity == WorldEventSeverity.CRITICAL:
            score = max(score, 0.75)
            reasons.append("Critical severity increases relevance.")

        score = self._apply_context_boost(
            score=score,
            event=event,
            context=context,
            reasons=reasons,
        )
        score = self._apply_confidence_and_verification(
            score=score,
            event=event,
            reasons=reasons,
        )
        score = self._clamp(score)
        should_alert = self._should_alert(event=event, relevance_score=score)

        if not reasons:
            reasons.append("No strong relevance signals were found.")

        return {
            "relevance_score": score,
            "reasons": reasons,
            "should_alert": should_alert,
        }

    def _score_weather_or_travel(
        self,
        event: WorldEvent,
        context: dict | None,
        reasons: list[str],
    ) -> float:
        """Score weather and aviation events."""
        interests = self._context_interests(context)
        if event.severity in {WorldEventSeverity.HIGH, WorldEventSeverity.CRITICAL}:
            reasons.append("High-severity weather/travel events can affect planning.")
            return 0.7

        if {"weather", "travel"} & interests:
            reasons.append("Context says weather or travel is currently relevant.")
            return 0.6

        reasons.append("Weather/travel event is not high severity.")
        return 0.4

    def _apply_context_boost(
        self,
        score: float,
        event: WorldEvent,
        context: dict | None,
        reasons: list[str],
    ) -> float:
        """Boost score when context interests match event text or tags."""
        interests = self._context_interests(context)
        event_terms = set(self._event_text(event).replace("/", " ").split())
        event_terms.update(tag.lower() for tag in event.tags)
        matched = interests & event_terms

        if matched:
            reasons.append(f"Context interests matched: {', '.join(sorted(matched))}.")
            return min(1.0, score + 0.12)

        return score

    def _apply_confidence_and_verification(
        self,
        score: float,
        event: WorldEvent,
        reasons: list[str],
    ) -> float:
        """Apply confidence and verification caps or boosts."""
        if event.verification_status in {VerificationStatus.DISPUTED, VerificationStatus.FALSE}:
            reasons.append("Disputed or false verification caps relevance.")
            return min(score, 0.2)

        if event.confidence_score < 0.3:
            reasons.append("Low confidence caps relevance.")
            return min(score, 0.4)

        if event.verification_status in {
            VerificationStatus.MULTI_SOURCE,
            VerificationStatus.TRUSTED_SOURCE,
        }:
            reasons.append("Strong verification gives a small boost.")
            return min(1.0, score + 0.05)

        return score

    def _should_alert(self, event: WorldEvent, relevance_score: float) -> bool:
        """Return True when the event should alert the user."""
        if relevance_score >= 0.85:
            return True

        if event.should_alert and relevance_score >= 0.6:
            return True

        return (
            event.severity == WorldEventSeverity.CRITICAL
            and event.confidence_score >= 0.5
        )

    def _context_interests(self, context: dict | None) -> set[str]:
        """Return lowercased context interests or default interests."""
        if context is None:
            return set(self.DEFAULT_INTERESTS)

        raw_interests = context.get("interests", self.DEFAULT_INTERESTS)
        return {str(interest).strip().lower() for interest in raw_interests}

    def _event_text(self, event: WorldEvent) -> str:
        """Return searchable event text."""
        values = [
            event.title,
            event.summary,
            event.category.value,
            event.severity.value,
            event.source_name or "",
            *event.tags,
        ]
        return " ".join(values).lower()

    def _clamp(self, score: float) -> float:
        """Keep a score in the valid 0.0 to 1.0 range."""
        return max(0.0, min(1.0, round(score, 2)))
