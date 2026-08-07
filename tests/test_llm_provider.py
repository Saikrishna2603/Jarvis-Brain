import pytest

from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_platform.schemas.llm import LLMMessage, LLMProviderName, LLMRequest, LLMResponse, LLMRole, LLMStatus


class FakeProvider(LLMProvider):
    name = LLMProviderName.MOCK
    model = "fake-model"

    def is_available(self) -> bool:
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(
            response_id="response-1",
            request_id=request.request_id,
            provider=self.name,
            model=self.model,
            status=LLMStatus.SUCCESS,
            content="fake",
        )


def test_base_provider_can_be_subclassed() -> None:
    provider = FakeProvider()
    request = LLMRequest(
        request_id="request-1",
        messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
    )

    response = provider.generate(request)

    assert response.content == "fake"


def test_is_available_can_return_bool() -> None:
    assert FakeProvider().is_available() is True


def test_base_generate_raises_not_implemented() -> None:
    provider = LLMProvider()
    request = LLMRequest(
        request_id="request-1",
        messages=[LLMMessage(role=LLMRole.USER, content="Hello")],
    )

    with pytest.raises(NotImplementedError):
        provider.generate(request)
