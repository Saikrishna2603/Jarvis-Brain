import pytest
from pydantic import ValidationError

from jarvis_platform.schemas.llm import (
    LLMMessage,
    LLMProviderName,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMStatus,
    LLMTaskType,
)


def test_can_create_llm_message() -> None:
    message = LLMMessage(role=LLMRole.USER, content="Hello")

    assert message.role == LLMRole.USER
    assert message.content == "Hello"


def test_empty_content_validation_works() -> None:
    with pytest.raises(ValidationError):
        LLMMessage(role=LLMRole.USER, content=" ")


def test_can_create_llm_request() -> None:
    request = LLMRequest(
        request_id="request-1",
        messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
    )

    assert request.provider == LLMProviderName.MOCK
    assert request.model == "mock-model"
    assert request.task_type == LLMTaskType.GENERAL


def test_empty_messages_validation_works() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(request_id="request-1", messages=[])


def test_temperature_validation_works() -> None:
    with pytest.raises(ValidationError):
        LLMRequest(
            request_id="request-1",
            messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
            temperature=2.5,
        )


def test_task_type_serializes_correctly() -> None:
    request = LLMRequest(
        request_id="request-1",
        task_type=LLMTaskType.CODING,
        messages=[LLMMessage(role=LLMRole.USER, content="Python error")],
    )

    assert request.model_dump(mode="json")["task_type"] == "coding"


def test_can_create_llm_response() -> None:
    response = LLMResponse(
        response_id="response-1",
        request_id="request-1",
        provider=LLMProviderName.MOCK,
        model="mock-model",
        status=LLMStatus.SUCCESS,
        content="Done",
    )

    assert response.content == "Done"


def test_is_success_works() -> None:
    response = LLMResponse(
        response_id="response-1",
        request_id="request-1",
        provider=LLMProviderName.MOCK,
        model="mock-model",
        status=LLMStatus.SUCCESS,
        content="Done",
    )

    assert response.is_success() is True


def test_is_error_works() -> None:
    response = LLMResponse(
        response_id="response-1",
        request_id="request-1",
        provider=LLMProviderName.MOCK,
        model="mock-model",
        status=LLMStatus.ERROR,
        content="",
    )

    assert response.is_error() is True


def test_serialization_works() -> None:
    response = LLMResponse(
        response_id="response-1",
        request_id="request-1",
        provider=LLMProviderName.MOCK,
        model="mock-model",
        status=LLMStatus.SUCCESS,
        content="Done",
    )

    dumped = response.model_dump(mode="json")

    assert dumped["provider"] == "mock"
    assert dumped["status"] == "success"
