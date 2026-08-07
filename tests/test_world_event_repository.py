import os

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from jarvis_platform.db.base import Base
from app.memory.world_event_repository import WorldEventRepository
from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory, WorldEventSeverity


pytestmark = pytest.mark.skipif(
    not os.getenv("TEST_DATABASE_URL"),
    reason="TEST_DATABASE_URL is not configured.",
)


@pytest.fixture()
def session():
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


def create_world_event(
    event_id: str = "event-1",
    category: WorldEventCategory = WorldEventCategory.CYBERSECURITY,
    severity: WorldEventSeverity = WorldEventSeverity.HIGH,
    relevance_score: float = 0.85,
    should_alert: bool = True,
) -> WorldEvent:
    """Create a world event for repository tests."""
    return WorldEvent(
        event_id=event_id,
        title="Mock cloud IAM advisory",
        summary="A mock advisory about cloud IAM misconfiguration risk.",
        category=category,
        severity=severity,
        confidence_score=0.75,
        relevance_score=relevance_score,
        should_alert=should_alert,
        tags=["cybersecurity", "cloud", "iam"],
        metadata={"source": "test"},
    )


def test_world_event_repository_can_save_and_get_event(session) -> None:
    repository = WorldEventRepository(session)
    event = create_world_event()

    saved = repository.save_event(event)
    found = repository.get_event(event.event_id)

    assert saved == event
    assert found == event
    assert found.metadata["source"] == "test"


def test_world_event_repository_can_save_multiple_events(session) -> None:
    repository = WorldEventRepository(session)
    events = [
        create_world_event(event_id="event-1"),
        create_world_event(event_id="event-2", category=WorldEventCategory.MARKETS),
    ]

    saved = repository.save_events(events)

    assert saved == events
    assert len(repository.get_all_events()) == 2


def test_world_event_repository_can_filter_by_category(session) -> None:
    repository = WorldEventRepository(session)
    cyber = repository.save_event(create_world_event(category=WorldEventCategory.CYBERSECURITY))
    repository.save_event(create_world_event(event_id="event-2", category=WorldEventCategory.MARKETS))

    events = repository.get_events_by_category("cybersecurity")

    assert events == [cyber]


def test_world_event_repository_can_get_high_priority_events(session) -> None:
    repository = WorldEventRepository(session)
    high_priority = repository.save_event(create_world_event())
    repository.save_event(
        create_world_event(
            event_id="event-2",
            severity=WorldEventSeverity.LOW,
            relevance_score=0.1,
            should_alert=False,
        )
    )

    events = repository.get_high_priority_events()

    assert events == [high_priority]
