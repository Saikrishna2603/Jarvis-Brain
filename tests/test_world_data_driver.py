import pytest

from jarvis_platform.schemas.world_event import WorldEventCategory
from app.tools.tool_registry import ToolRegistry
from app.tools.world_data_driver import WorldDataDriver


def test_world_data_driver_has_name_world_data() -> None:
    driver = WorldDataDriver()

    assert driver.name == "world_data"


def test_supported_actions_contains_get_global_briefing() -> None:
    driver = WorldDataDriver()

    assert "get_global_briefing" in driver.supported_actions


def test_can_handle_returns_true_for_supported_action() -> None:
    driver = WorldDataDriver()

    assert driver.can_handle("get_cyber_events") is True


def test_can_handle_returns_false_for_unsupported_action() -> None:
    driver = WorldDataDriver()

    assert driver.can_handle("send_email") is False


def test_get_global_briefing_returns_mixed_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_global_briefing()
    categories = {event.category for event in events}

    assert len(events) == 7
    assert WorldEventCategory.CYBERSECURITY in categories
    assert WorldEventCategory.MARKETS in categories
    assert WorldEventCategory.WEATHER in categories
    assert WorldEventCategory.AI_RESEARCH in categories
    assert WorldEventCategory.GEOPOLITICS in categories
    assert WorldEventCategory.AVIATION in categories
    assert WorldEventCategory.ENERGY in categories


def test_get_cyber_events_returns_cybersecurity_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_cyber_events()

    assert events
    assert all(event.category == WorldEventCategory.CYBERSECURITY for event in events)


def test_get_market_events_returns_market_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_market_events()

    assert events
    assert all(event.category == WorldEventCategory.MARKETS for event in events)


def test_get_weather_events_returns_weather_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_weather_events()

    assert events
    assert all(event.category == WorldEventCategory.WEATHER for event in events)


def test_get_ai_research_events_returns_ai_research_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_ai_research_events()

    assert events
    assert all(event.category == WorldEventCategory.AI_RESEARCH for event in events)


def test_get_geopolitical_events_returns_geopolitical_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_geopolitical_events()

    assert events
    assert all(event.category == WorldEventCategory.GEOPOLITICS for event in events)


def test_get_aviation_events_returns_aviation_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_aviation_events()

    assert events
    assert all(event.category == WorldEventCategory.AVIATION for event in events)


def test_get_energy_events_returns_energy_events() -> None:
    driver = WorldDataDriver()

    events = driver.get_energy_events()

    assert events
    assert all(event.category == WorldEventCategory.ENERGY for event in events)


def test_search_world_events_filters_by_title() -> None:
    driver = WorldDataDriver()

    events = driver.search_world_events("cloud IAM")

    assert len(events) == 1
    assert events[0].title == "Mock cloud IAM advisory"


def test_search_world_events_filters_by_tag() -> None:
    driver = WorldDataDriver()

    events = driver.search_world_events("frameworks")

    assert len(events) == 1
    assert events[0].category == WorldEventCategory.AI_RESEARCH


def test_execute_returns_success_dict_for_get_global_briefing() -> None:
    driver = WorldDataDriver()

    result = driver.execute("get_global_briefing")

    assert result["status"] == "success"
    assert result["action"] == "get_global_briefing"
    assert result["source"] == "world_data"
    assert result["count"] == 7
    assert result["events"][0]["event_id"]


def test_execute_returns_success_dict_for_search_world_events() -> None:
    driver = WorldDataDriver()

    result = driver.execute("search_world_events", payload={"query": "aviation"})

    assert result["status"] == "success"
    assert result["action"] == "search_world_events"
    assert result["count"] == 1
    assert result["events"][0]["category"] == "aviation"


def test_execute_raises_value_error_for_unknown_action() -> None:
    driver = WorldDataDriver()

    with pytest.raises(ValueError, match="WorldDataDriver cannot handle action"):
        driver.execute("unknown_action")


def test_driver_works_through_tool_registry() -> None:
    registry = ToolRegistry()
    registry.register_driver(WorldDataDriver())

    result = registry.execute_action("get_weather_events")

    assert result["status"] == "success"
    assert result["source"] == "world_data"
    assert result["events"][0]["category"] == "weather"


def test_high_priority_cyber_event_returns_true() -> None:
    driver = WorldDataDriver()

    cyber_event = driver.get_cyber_events()[0]

    assert cyber_event.is_high_priority() is True


def test_all_returned_events_serialize_correctly() -> None:
    driver = WorldDataDriver()

    events = driver.get_global_briefing()

    for event in events:
        dumped = event.model_dump(mode="json")
        json_text = event.model_dump_json()

        assert dumped["event_id"]
        assert dumped["title"]
        assert isinstance(json_text, str)
