from jarvis_platform.schemas.llm import LLMRequest, LLMResponse


class LLMConsensusEngine:
    """Foundation for future multi-model consensus.

    Phase 4 intentionally does not run debates or multiple model calls.
    """

    def propose(self, _request: LLMRequest) -> list[LLMResponse]:
        return []

    def validate_consensus(self, responses: list[LLMResponse]) -> LLMResponse | None:
        return responses[0] if responses else None
