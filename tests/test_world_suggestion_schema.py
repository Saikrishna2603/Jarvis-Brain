import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.world_suggestion import (
    SuggestionPriority,
    SuggestionStatus,
    SuggestionType,
    WorldSuggestion,
)


def test_can_create_world_suggestion() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Review security event",
        message="Review security controls.",
    )

    assert suggestion.suggestion_id == "suggestion-1"
    assert suggestion.suggestion_type == SuggestionType.UNKNOWN
    assert suggestion.priority == SuggestionPriority.LOW
    assert suggestion.status == SuggestionStatus.PENDING
    assert suggestion.metadata == {}
    assert suggestion.created_at.tzinfo is not None


def test_empty_title_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldSuggestion(
            suggestion_id="suggestion-1",
            title=" ",
            message="Valid message.",
        )


def test_empty_message_validation_works() -> None:
    with pytest.raises(ValidationError):
        WorldSuggestion(
            suggestion_id="suggestion-1",
            title="Valid title",
            message="",
        )


def test_is_high_priority_returns_true_for_high() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="High priority",
        message="Review this.",
        priority=SuggestionPriority.HIGH,
    )

    assert suggestion.is_high_priority() is True


def test_is_high_priority_returns_true_for_urgent() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Urgent priority",
        message="Review this now.",
        priority=SuggestionPriority.URGENT,
    )

    assert suggestion.is_high_priority() is True


def test_needs_user_response_returns_true_for_requires_user_approval() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Needs approval",
        message="Please approve.",
        requires_user_approval=True,
    )

    assert suggestion.needs_user_response() is True


def test_needs_user_response_returns_true_for_ask_user() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Ask user",
        message="Ask the user.",
        suggestion_type=SuggestionType.ASK_USER,
    )

    assert suggestion.needs_user_response() is True


def test_needs_user_response_returns_true_for_escalate_alert() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Escalate",
        message="Alert the user.",
        suggestion_type=SuggestionType.ESCALATE_ALERT,
    )

    assert suggestion.needs_user_response() is True


def test_schema_serializes_correctly() -> None:
    suggestion = WorldSuggestion(
        suggestion_id="suggestion-1",
        title="Serializable",
        message="This can be serialized.",
        suggestion_type=SuggestionType.REVIEW_RESEARCH,
        metadata={"source": "test"},
    )

    dumped = suggestion.model_dump()
    json_text = suggestion.model_dump_json()

    assert dumped["suggestion_id"] == "suggestion-1"
    assert dumped["metadata"] == {"source": "test"}
    assert '"suggestion_type":"review_research"' in json_text
