from collections.abc import Iterator

from jarvis_platform.schemas.llm import LLMProviderName, LLMRequest, LLMResponse


class LLMProvider:
    """Base interface for LLM providers."""

    name: LLMProviderName = LLMProviderName.UNKNOWN
    model: str = "unknown"

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response for the request.

        Subclasses must implement this method.
        """
        raise NotImplementedError("LLM providers must implement generate().")

    def is_available(self) -> bool:
        """Return whether the provider can currently accept requests."""
        return False

    def generate_stream(self, request: LLMRequest) -> Iterator[dict]:
        """Yield provider-neutral token events when streaming is supported."""
        response = self.generate(request)
        if response.is_success() and response.content:
            yield {"type": "token", "content": response.content}
            yield {"type": "done", "metadata": response.raw_metadata}
            return
        yield {
            "type": "error",
            "message": response.error_message or "The LLM provider is unavailable.",
        }
