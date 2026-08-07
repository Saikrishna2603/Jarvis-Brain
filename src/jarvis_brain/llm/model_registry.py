import os

from jarvis_platform.config import load_app_environment
from jarvis_platform.schemas.llm import LLMProviderName, LLMTaskType
from jarvis_platform.schemas.llm_router import LLMModelCapability


class LLMModelRegistry:
    """Safe public model capability registry."""

    def __init__(self) -> None:
        load_app_environment()
        general_model = os.getenv("LLM_GENERAL_MODEL", "llama3.1:8b")
        coding_model = os.getenv("LLM_CODING_MODEL", "qwen2.5-coder:7b")
        classification_model = os.getenv(
            "LLM_CLASSIFICATION_MODEL", "qwen2.5-coder:3b"
        )
        self.models = [
            LLMModelCapability(
                model="mock-model",
                provider=LLMProviderName.MOCK,
                task_types=list(LLMTaskType),
                context_length=4096,
                supports_structured_output=True,
                reasoning_score=1,
                coding_score=1,
            ),
            LLMModelCapability(
                model=general_model,
                provider=LLMProviderName.OLLAMA,
                task_types=[
                    LLMTaskType.CONVERSATION,
                    LLMTaskType.GENERAL,
                    LLMTaskType.REASONING,
                    LLMTaskType.PLANNING,
                    LLMTaskType.SUMMARIZATION,
                    LLMTaskType.GUIDANCE,
                    LLMTaskType.WORLD_INTELLIGENCE,
                ],
                context_length=8192,
                supports_streaming=True,
                reasoning_score=3,
                coding_score=2,
            ),
            LLMModelCapability(
                model=coding_model,
                provider=LLMProviderName.OLLAMA,
                task_types=[LLMTaskType.CODING, LLMTaskType.STRUCTURED_EXTRACTION, LLMTaskType.TOOL_PLANNING],
                context_length=8192,
                supports_streaming=True,
                supports_structured_output=True,
                reasoning_score=2,
                coding_score=4,
            ),
            LLMModelCapability(
                model=classification_model,
                provider=LLMProviderName.OLLAMA,
                task_types=[
                    LLMTaskType.INTENT_RESOLUTION,
                    LLMTaskType.STRUCTURED_EXTRACTION,
                ],
                context_length=8192,
                supports_streaming=True,
                supports_structured_output=True,
                reasoning_score=2,
                coding_score=3,
            ),
        ]

    def list_models(self) -> list[LLMModelCapability]:
        return list(self.models)

    def select_model(self, provider: LLMProviderName, task_type: LLMTaskType) -> str:
        exact_candidates = [
            model for model in self.models
            if model.provider == provider and task_type in model.task_types
        ]
        candidates = exact_candidates or [
            model for model in self.models
            if model.provider == provider and LLMTaskType.GENERAL in model.task_types
        ]
        if not candidates:
            fallback = next((model for model in self.models if model.provider == provider), None)
            return fallback.model if fallback else "mock-model"
        if task_type == LLMTaskType.CODING:
            return max(candidates, key=lambda item: item.coding_score).model
        return max(candidates, key=lambda item: item.reasoning_score).model
