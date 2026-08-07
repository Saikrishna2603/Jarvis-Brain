import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from jarvis_platform.db.base import Base
from jarvis_platform.schemas.world_impact import WorldImpactAnalysis
from jarvis_platform.schemas.world_suggestion import SuggestionPriority, SuggestionType, WorldSuggestion
from jarvis_brain.world.proactive_event_loop import ProactiveEventLoop
from jarvis_brain.world.world_intelligence_repository import WorldIntelligenceRepository


@pytest.fixture()
def session():
    if not os.getenv("TEST_DATABASE_URL"):
        pytest.skip("TEST_DATABASE_URL is not configured.")

    engine = create_engine(os.environ["TEST_DATABASE_URL"])
    with engine.begin() as connection:
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS world"))
        connection.execute(text("CREATE SCHEMA IF NOT EXISTS knowledge"))
    Base.metadata.create_all(bind=engine)
    SessionLocal = sessionmaker(bind=engine)
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
        Base.metadata.drop_all(bind=engine)


def create_analysis(analysis_id: str = "analysis-1") -> WorldImpactAnalysis:
    """Create an impact analysis for repository tests."""
    return WorldImpactAnalysis(
        analysis_id=analysis_id,
        event_id="event-1",
        event_title="Mock cloud IAM advisory",
        category="cybersecurity",
        severity="high",
        relevance_score=0.9,
        impact_summary="This may affect Jarvis security posture.",
        affected_areas=["security", "cloud"],
        suggested_actions=["Review security controls."],
        risk_level="high",
        confidence_score=0.75,
        requires_user_discussion=True,
        should_create_task=True,
        metadata={"source": "test"},
    )


def create_suggestion(suggestion_id: str = "suggestion-1") -> WorldSuggestion:
    """Create a world suggestion for repository tests."""
    return WorldSuggestion(
        suggestion_id=suggestion_id,
        event_id="event-1",
        analysis_id="analysis-1",
        title="Review security event",
        message="Review security controls.",
        suggestion_type=SuggestionType.REVIEW_SECURITY,
        priority=SuggestionPriority.HIGH,
        metadata={"source": "test"},
    )


def test_world_intelligence_repository_can_save_and_get_analyses(session) -> None:
    repository = WorldIntelligenceRepository(session)
    analysis = create_analysis()

    saved = repository.save_analysis(analysis)
    all_analyses = repository.get_all_analyses()
    event_analyses = repository.get_analyses_by_event("event-1")

    assert saved == analysis
    assert all_analyses == [analysis]
    assert event_analyses == [analysis]


def test_world_intelligence_repository_can_save_and_get_suggestions(session) -> None:
    repository = WorldIntelligenceRepository(session)
    suggestion = create_suggestion()

    saved = repository.save_suggestion(suggestion)
    all_suggestions = repository.get_all_suggestions()
    event_suggestions = repository.get_suggestions_by_event("event-1")

    assert saved == suggestion
    assert all_suggestions == [suggestion]
    assert event_suggestions == [suggestion]


def test_world_intelligence_repository_can_get_alert_suggestions(session) -> None:
    repository = WorldIntelligenceRepository(session)
    alert = repository.save_suggestion(create_suggestion())
    repository.save_suggestion(
        create_suggestion(suggestion_id="suggestion-2").model_copy(
            update={"priority": SuggestionPriority.LOW}
        )
    )

    alerts = repository.get_alert_suggestions()

    assert alerts == [alert]


def test_proactive_event_loop_calls_fake_repositories() -> None:
    class FakeWorldEventRepository:
        def __init__(self) -> None:
            self.saved_events = []

        def save_events(self, events):
            self.saved_events = events
            return events

    class FakeWorldIntelligenceRepository:
        def __init__(self) -> None:
            self.saved_analyses = []
            self.saved_suggestions = []

        def save_analyses(self, analyses):
            self.saved_analyses = analyses
            return analyses

        def save_suggestions(self, suggestions):
            self.saved_suggestions = suggestions
            return suggestions

    event_repository = FakeWorldEventRepository()
    intelligence_repository = FakeWorldIntelligenceRepository()
    loop = ProactiveEventLoop(
        world_event_repository=event_repository,
        world_intelligence_repository=intelligence_repository,
    )

    result = loop.run_once()

    assert result["status"] == "success"
    assert event_repository.saved_events
    assert intelligence_repository.saved_analyses
    assert intelligence_repository.saved_suggestions
