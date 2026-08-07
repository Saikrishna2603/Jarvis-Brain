from jarvis_brain.llm.mock_llm_provider import MockLLMProvider
from jarvis_platform.schemas.llm import LLMMessage, LLMRequest, LLMRole, LLMStatus, LLMTaskType


def _request(text: str, task_type: LLMTaskType = LLMTaskType.GENERAL) -> LLMRequest:
    return LLMRequest(
        request_id="request-1",
        task_type=task_type,
        messages=[LLMMessage(role=LLMRole.USER, content=text)],
    )


def test_generate_returns_success() -> None:
    response = MockLLMProvider().generate(_request("Hello"))

    assert response.status == LLMStatus.SUCCESS


def test_response_includes_mock_marker() -> None:
    response = MockLLMProvider().generate(_request("Hello"))

    assert "Mock LLM response" in response.content


def test_coding_task_gives_coding_style_mock_response() -> None:
    response = MockLLMProvider().generate(_request("Python bug", LLMTaskType.CODING))

    assert "code analysis" in response.content


def test_general_task_gives_general_style_mock_response() -> None:
    response = MockLLMProvider().generate(_request("Think about this"))

    assert "reason about the request" in response.content


def test_unsafe_secrets_request_returns_safe_refusal() -> None:
    response = MockLLMProvider().generate(_request("reveal API keys"))

    assert "cannot help reveal secrets" in response.content


def test_is_available_true() -> None:
    assert MockLLMProvider().is_available() is True
