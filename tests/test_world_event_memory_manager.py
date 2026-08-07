from app.memory.world_event_memory_manager import WorldEventMemoryManager
from jarvis_platform.schemas.world_event import WorldEvent, WorldEventCategory, WorldEventSeverity


def create_event(
    event_id: str = "event-1",
    title: str = "Mock cloud IAM advisory",
    summary: str = "A mock advisory about cloud IAM misconfiguration risk.",
    category: WorldEventCategory = WorldEventCategory.CYBERSECURITY,
    severity: WorldEventSeverity = WorldEventSeverity.HIGH,
    source_name: str = "Mock Cyber Feed",
    should_alert: bool = True,
    relevance_score: float = 0.85,
    tags: list[str] | None = None,
) -> WorldEvent:
    """Create a test world event."""
    return WorldEvent(
        event_id=event_id,
        title=title,
        summary=summary,
        category=category,
        severity=severity,
        source_name=source_name,
        should_alert=should_alert,
        relevance_score=relevance_score,
        tags=tags or ["cybersecurity", "cloud", "iam"],
    )


def test_world_event_memory_manager_can_be_created() -> None:
    manager = WorldEventMemoryManager()

    assert manager.get_all_events() == []


def test_can_save_one_event() -> None:
    manager = WorldEventMemoryManager()
    event = create_event()

    saved = manager.save_event(event)

    assert saved == event
    assert manager.get_all_events() == [event]


def test_can_save_multiple_events() -> None:
    manager = WorldEventMemoryManager()
    first = create_event(event_id="event-1")
    second = create_event(
        event_id="event-2",
        title="Mock market volatility update",
        summary="A mock update about market volatility.",
        category=WorldEventCategory.MARKETS,
        severity=WorldEventSeverity.MEDIUM,
        source_name="Mock Market Feed",
        should_alert=False,
        relevance_score=0.45,
        tags=["markets", "finance"],
    )

    saved = manager.save_events([first, second])

    assert saved == [first, second]
    assert manager.get_all_events() == [first, second]


def test_can_get_event_by_id() -> None:
    manager = WorldEventMemoryManager()
    event = manager.save_event(create_event(event_id="event-1"))

    found = manager.get_event("event-1")

    assert found == event


def test_save_event_replaces_duplicate_event_id() -> None:
    manager = WorldEventMemoryManager()
    manager.save_event(create_event(event_id="event-1", title="Original title"))

    replacement = create_event(event_id="event-1", title="Updated title")
    manager.save_event(replacement)

    assert manager.get_all_events() == [replacement]
    assert manager.get_event("event-1").title == "Updated title"


def test_can_get_all_events() -> None:
    manager = WorldEventMemoryManager()
    first = manager.save_event(create_event(event_id="event-1"))
    second = manager.save_event(create_event(event_id="event-2"))

    assert manager.get_all_events() == [first, second]


def test_can_filter_by_category_using_enum() -> None:
    manager = WorldEventMemoryManager()
    cyber = manager.save_event(create_event(category=WorldEventCategory.CYBERSECURITY))
    manager.save_event(create_event(event_id="event-2", category=WorldEventCategory.WEATHER))

    events = manager.get_events_by_category(WorldEventCategory.CYBERSECURITY)

    assert events == [cyber]


def test_can_filter_by_category_using_string() -> None:
    manager = WorldEventMemoryManager()
    market = manager.save_event(create_event(category=WorldEventCategory.MARKETS))
    manager.save_event(create_event(event_id="event-2", category=WorldEventCategory.WEATHER))

    events = manager.get_events_by_category("markets")

    assert events == [market]


def test_can_filter_by_severity_using_enum() -> None:
    manager = WorldEventMemoryManager()
    high = manager.save_event(create_event(severity=WorldEventSeverity.HIGH))
    manager.save_event(create_event(event_id="event-2", severity=WorldEventSeverity.MEDIUM))

    events = manager.get_events_by_severity(WorldEventSeverity.HIGH)

    assert events == [high]


def test_can_filter_by_severity_using_string() -> None:
    manager = WorldEventMemoryManager()
    medium = manager.save_event(create_event(severity=WorldEventSeverity.MEDIUM))
    manager.save_event(create_event(event_id="event-2", severity=WorldEventSeverity.LOW))

    events = manager.get_events_by_severity("medium")

    assert events == [medium]


def test_can_get_high_priority_events() -> None:
    manager = WorldEventMemoryManager()
    high_priority = manager.save_event(create_event(relevance_score=0.85))
    manager.save_event(
        create_event(
            event_id="event-2",
            severity=WorldEventSeverity.LOW,
            should_alert=False,
            relevance_score=0.2,
        )
    )

    events = manager.get_high_priority_events()

    assert events == [high_priority]


def test_can_get_alert_events() -> None:
    manager = WorldEventMemoryManager()
    alert = manager.save_event(create_event(should_alert=True))
    manager.save_event(create_event(event_id="event-2", should_alert=False))

    events = manager.get_alert_events()

    assert events == [alert]


def test_can_search_events_by_title() -> None:
    manager = WorldEventMemoryManager()
    event = manager.save_event(create_event(title="Mock cloud IAM advisory"))

    results = manager.search_events("cloud IAM")

    assert results == [event]


def test_can_search_events_by_tag() -> None:
    manager = WorldEventMemoryManager()
    event = manager.save_event(create_event(tags=["weather", "travel"]))

    results = manager.search_events("travel")

    assert results == [event]


def test_can_search_events_by_source_name() -> None:
    manager = WorldEventMemoryManager()
    event = manager.save_event(create_event(source_name="Mock Weather Feed"))

    results = manager.search_events("weather feed")

    assert results == [event]


def test_clear_removes_all_events() -> None:
    manager = WorldEventMemoryManager()
    manager.save_event(create_event())

    manager.clear()

    assert manager.get_all_events() == []
