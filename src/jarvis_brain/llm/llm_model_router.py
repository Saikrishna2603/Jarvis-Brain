import re

from jarvis_platform.schemas.llm import LLMTaskType


class LLMModelRouter:
    """Route LLM tasks to the safest local model for the job."""

    def __init__(
        self,
        general_model: str = "llama3.1:8b",
        coding_model: str = "qwen2.5-coder:7b",
        classification_model: str = "qwen2.5-coder:3b",
        default_model: str = "llama3.1:8b",
    ) -> None:
        """Create a model router with configured model names."""
        self.general_model = general_model
        self.coding_model = coding_model
        self.classification_model = classification_model
        self.default_model = default_model

    def detect_task_type(
        self,
        text: str,
        metadata: dict | None = None,
    ) -> LLMTaskType:
        """Detect the likely task type from text and optional metadata."""
        metadata = metadata or {}
        override = metadata.get("task_type")
        if isinstance(override, LLMTaskType):
            return override
        if isinstance(override, str):
            try:
                return LLMTaskType(override)
            except ValueError:
                return LLMTaskType.UNKNOWN

        normalized = text.lower()
        coding_keywords = {
            "code",
            "python",
            "javascript",
            "java",
            "fastapi",
            "react",
            "docker",
            "kubernetes",
            "sql",
            "traceback",
            "error",
            "exception",
            "repository",
            "repo",
            "function",
            "class",
            "pytest",
            "bug",
            "api endpoint",
            "alembic",
            "sqlalchemy",
        }
        if any(_contains_keyword(normalized, keyword) for keyword in coding_keywords):
            return LLMTaskType.CODING

        if any(keyword in normalized for keyword in {"plan", "roadmap", "steps", "architecture", "design"}):
            return LLMTaskType.PLANNING

        if any(keyword in normalized for keyword in {"summarize", "summary", "explain this text"}):
            return LLMTaskType.SUMMARIZATION

        if any(keyword in normalized for keyword in {"classify", "extract fields", "detect intent"}):
            return LLMTaskType.INTENT_RESOLUTION

        if any(keyword in normalized for keyword in {"how do i", "troubleshoot", "fix", "guide me", "step by step"}):
            return LLMTaskType.GUIDANCE

        return LLMTaskType.GENERAL

    def select_model(self, task_type: LLMTaskType) -> str:
        """Return the model name for the task type."""
        if task_type == LLMTaskType.CODING:
            return self.coding_model
        if task_type in {
            LLMTaskType.INTENT_RESOLUTION,
            LLMTaskType.STRUCTURED_EXTRACTION,
        }:
            return self.classification_model
        if task_type in {
            LLMTaskType.GENERAL,
            LLMTaskType.PLANNING,
            LLMTaskType.SUMMARIZATION,
            LLMTaskType.GUIDANCE,
        }:
            return self.general_model
        return self.default_model

    def route(self, text: str, metadata: dict | None = None) -> dict[str, str]:
        """Return task type and selected model for a text input."""
        task_type = self.detect_task_type(text, metadata)
        return {
            "task_type": task_type.value,
            "model": self.select_model(task_type),
        }


def _contains_keyword(text: str, keyword: str) -> bool:
    return re.search(rf"(?<!\w){re.escape(keyword)}(?!\w)", text) is not None
