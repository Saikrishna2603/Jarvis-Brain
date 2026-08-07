import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.world_event import (
    SourceVisibility,
    VerificationStatus,
    WorldEvent,
    WorldEventCategory,
    WorldEventSeverity,
)


def test_can_create_world_event_with_defaults() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="General public update",
        summary="A public update was collected.",
    )

    assert event.category == WorldEventCategory.UNKNOWN
    assert event.severity == WorldEventSeverity.LOW
    assert event.source_visibility == SourceVisibility.UNKNOWN
    assert event.verification_status == VerificationStatus.UNVERIFIED
    assert event.confidence_score == 0.0
    assert event.relevance_score == 0.0
    assert event.tags == []
    assert event.metadata == {}
    assert event.collected_at.tzinfo is not None


def test_can_create_cybersecurity_event() -> None:
    event = WorldEvent(
        event_id="event-cyber-1",
        title="Critical vulnerability advisory",
        summary="A vendor disclosed a new critical vulnerability.",
        category=WorldEventCategory.CYBERSECURITY,
        severity=WorldEventSeverity.HIGH,
        tags=["cve", "security"],
    )

    assert event.category == WorldEventCategory.CYBERSECURITY
    assert event.severity == WorldEventSeverity.HIGH
    assert event.tags == ["cve", "security"]


def test_can_create_public_source_event() -> None:
    event = WorldEvent(
        event_id="event-public-1",
        title="Public news report",
        summary="A public news source reported an event.",
        source_name="Example News",
        source_url="https://example.com/news",
        source_visibility=SourceVisibility.PUBLIC,
    )

    assert event.source_name == "Example News"
    assert event.source_url == "https://example.com/news"
    assert event.source_visibility == SourceVisibility.PUBLIC


def test_confidence_score_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldEvent(
            event_id="event-1",
            title="Invalid confidence",
            summary="The confidence score is invalid.",
            confidence_score=1.5,
        )


def test_relevance_score_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldEvent(
            event_id="event-1",
            title="Invalid relevance",
            summary="The relevance score is invalid.",
            relevance_score=-0.1,
        )


def test_empty_title_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldEvent(
            event_id="event-1",
            title="   ",
            summary="The title is empty.",
        )


def test_empty_summary_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldEvent(
            event_id="event-1",
            title="Empty summary",
            summary="",
        )


def test_is_high_priority_returns_true_for_high_severity() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="High severity event",
        summary="A high severity event happened.",
        severity=WorldEventSeverity.HIGH,
    )

    assert event.is_high_priority() is True


def test_is_high_priority_returns_true_for_critical_severity() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Critical event",
        summary="A critical event happened.",
        severity=WorldEventSeverity.CRITICAL,
    )

    assert event.is_high_priority() is True


def test_is_high_priority_returns_true_for_should_alert() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Alert event",
        summary="An event should alert the user.",
        should_alert=True,
    )

    assert event.is_high_priority() is True


def test_is_high_priority_returns_true_for_relevance_score() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Relevant event",
        summary="A highly relevant event happened.",
        relevance_score=0.8,
    )

    assert event.is_high_priority() is True


def test_is_verified_returns_true_for_multi_source() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Multi-source event",
        summary="Multiple sources confirmed this event.",
        verification_status=VerificationStatus.MULTI_SOURCE,
    )

    assert event.is_verified() is True


def test_is_verified_returns_true_for_trusted_source() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Trusted-source event",
        summary="A trusted source confirmed this event.",
        verification_status=VerificationStatus.TRUSTED_SOURCE,
    )

    assert event.is_verified() is True


def test_is_public_source_returns_true_for_public_source() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Public event",
        summary="This event came from a public source.",
        source_visibility=SourceVisibility.PUBLIC,
    )

    assert event.is_public_source() is True


def test_world_event_serializes_correctly() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="Market event",
        summary="A market event happened.",
        category=WorldEventCategory.MARKETS,
        metadata={"symbol": "SPY"},
    )

    dumped = event.model_dump()
    json_text = event.model_dump_json()

    assert dumped["event_id"] == "event-1"
    assert dumped["metadata"] == {"symbol": "SPY"}
    assert isinstance(json_text, str)


def test_enums_serialize_correctly() -> None:
    event = WorldEvent(
        event_id="event-1",
        title="AI research event",
        summary="A new public AI research item was collected.",
        category=WorldEventCategory.AI_RESEARCH,
        severity=WorldEventSeverity.MEDIUM,
        verification_status=VerificationStatus.SINGLE_SOURCE,
        source_visibility=SourceVisibility.PUBLIC,
    )

    dumped_json = event.model_dump_json()

    assert '"category":"ai_research"' in dumped_json
    assert '"severity":"medium"' in dumped_json
    assert '"verification_status":"single_source"' in dumped_json
    assert '"source_visibility":"public"' in dumped_json
