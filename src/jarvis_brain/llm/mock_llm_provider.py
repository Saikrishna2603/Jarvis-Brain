from uuid import uuid4

from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_platform.schemas.llm import (
    LLMProviderName,
    LLMRequest,
    LLMResponse,
    LLMStatus,
    LLMTaskType,
)


class MockLLMProvider(LLMProvider):
    """Deterministic local provider used when real LLMs are disabled."""

    name = LLMProviderName.MOCK
    model = "mock-model"

    def is_available(self) -> bool:
        """Mock provider is always available."""
        return True

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Return a deterministic safe response for tests and local defaults."""
        joined_content = " ".join(message.content for message in request.messages).lower()
        if self._looks_unsafe(joined_content):
            content = "Mock LLM response: I cannot help reveal secrets or bypass safety."
        elif request.task_type == LLMTaskType.CODING:
            content = "Mock LLM response: I would help with code analysis here."
        else:
            content = "Mock LLM response: I would reason about the request here."

        return LLMResponse(
            response_id=str(uuid4()),
            request_id=request.request_id,
            provider=self.name,
            model=self.model,
            task_type=request.task_type,
            status=LLMStatus.SUCCESS,
            content=content,
            raw_metadata={"mock": True},
        )

    def _looks_unsafe(self, text: str) -> bool:
        """Return True for simple secret-exfiltration or bypass wording."""
        unsafe_phrases = {
            "reveal secrets",
            "reveal api keys",
            "show api keys",
            "system prompt",
            "bypass safety",
            "bypass security",
            "ignore previous instructions",
        }
        return any(phrase in text for phrase in unsafe_phrases)
