import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.world_impact import WorldImpactAnalysis


def test_can_create_world_impact_analysis() -> None:
    analysis = WorldImpactAnalysis(
        analysis_id="analysis-1",
        event_id="event-1",
        event_title="Mock cloud IAM advisory",
        category="cybersecurity",
        severity="high",
        impact_summary="This may affect Jarvis security posture.",
    )

    assert analysis.analysis_id == "analysis-1"
    assert analysis.relevance_score == 0.0
    assert analysis.risk_level == "low"
    assert analysis.affected_areas == []
    assert analysis.created_at.tzinfo is not None


def test_empty_impact_summary_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldImpactAnalysis(
            analysis_id="analysis-1",
            event_id="event-1",
            event_title="Empty summary",
            category="news",
            severity="low",
            impact_summary=" ",
        )


def test_relevance_score_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldImpactAnalysis(
            analysis_id="analysis-1",
            event_id="event-1",
            event_title="Bad relevance",
            category="news",
            severity="low",
            relevance_score=1.2,
            impact_summary="Invalid relevance score.",
        )


def test_confidence_score_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldImpactAnalysis(
            analysis_id="analysis-1",
            event_id="event-1",
            event_title="Bad confidence",
            category="news",
            severity="low",
            confidence_score=-0.1,
            impact_summary="Invalid confidence score.",
        )


def test_is_actionable_returns_true_when_suggested_actions_exists() -> None:
    analysis = WorldImpactAnalysis(
        analysis_id="analysis-1",
        event_id="event-1",
        event_title="Actionable",
        category="cybersecurity",
        severity="high",
        impact_summary="Action may be useful.",
        suggested_actions=["Review security controls."],
    )

    assert analysis.is_actionable() is True


def test_is_actionable_returns_true_when_requires_user_discussion() -> None:
    analysis = WorldImpactAnalysis(
        analysis_id="analysis-1",
        event_id="event-1",
        event_title="Discussion",
        category="news",
        severity="low",
        impact_summary="Discussion may be useful.",
        requires_user_discussion=True,
    )

    assert analysis.is_actionable() is True


def test_schema_serializes_correctly() -> None:
    analysis = WorldImpactAnalysis(
        analysis_id="analysis-1",
        event_id="event-1",
        event_title="Serializable",
        category="ai_research",
        severity="medium",
        impact_summary="This can be serialized.",
        metadata={"source": "test"},
    )

    dumped = analysis.model_dump()
    json_text = analysis.model_dump_json()

    assert dumped["analysis_id"] == "analysis-1"
    assert dumped["metadata"] == {"source": "test"}
    assert isinstance(json_text, str)
