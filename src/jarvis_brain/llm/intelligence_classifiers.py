from jarvis_platform.schemas.llm import LLMMessage, LLMRole, LLMTaskType
from jarvis_platform.schemas.llm_router import LLMPrivacyClass


class CapabilityClassifier:
    """Deterministically classify LLM work into capability buckets."""

    def classify(self, messages: list[LLMMessage], metadata: dict | None = None) -> LLMTaskType:
        metadata = metadata or {}
        override = metadata.get("task_type")
        if isinstance(override, str):
            try:
                return LLMTaskType(override)
            except ValueError:
                pass

        text = _joined_user_text(messages)
        checks: list[tuple[LLMTaskType, tuple[str, ...]]] = [
            (LLMTaskType.INTENT_RESOLUTION, ("classify", "classification", "intent category")),
            (LLMTaskType.CODING, ("code", "python", "fastapi", "traceback", "pytest", "sqlalchemy", "repo", "bug", "function", "class")),
            (LLMTaskType.PLANNING, ("plan", "roadmap", "architecture", "steps", "design")),
            (LLMTaskType.WORLD_INTELLIGENCE, ("world briefing", "cyber alerts", "world intelligence", "global update")),
            (LLMTaskType.VISION, ("image", "screenshot", "visual", "photo", "diagram")),
            (LLMTaskType.VOICE, ("voice", "speech", "tts", "stt", "speak")),
            (LLMTaskType.RETRIEVAL, ("retrieve", "source", "evidence", "citation", "official docs")),
            (LLMTaskType.STRUCTURED_EXTRACTION, ("extract json", "structured", "parse this", "schema")),
            (LLMTaskType.SUMMARIZATION, ("summarize", "summary", "brief this")),
            (LLMTaskType.TOOL_PLANNING, ("tool", "integration", "connector", "api action")),
            (LLMTaskType.CREATIVE, ("write a story", "brainstorm", "creative", "tagline")),
            (LLMTaskType.REASONING, ("why", "reason", "compare", "decide", "analyze")),
        ]
        for task_type, keywords in checks:
            if any(keyword in text for keyword in keywords):
                return task_type
        return LLMTaskType.CONVERSATION


class PrivacyClassifier:
    """Classify privacy routing without using an LLM."""

    sensitive_keywords = (
        "api key",
        "password",
        "token",
        "secret",
        "private key",
        "credential",
        "ssn",
        "medical",
        "legal",
        "bank",
        "finance account",
    )

    def classify(self, messages: list[LLMMessage], metadata: dict | None = None) -> LLMPrivacyClass:
        metadata = metadata or {}
        override = metadata.get("privacy_class")
        if isinstance(override, str):
            try:
                return LLMPrivacyClass(override)
            except ValueError:
                pass
        text = _joined_user_text(messages)
        if any(keyword in text for keyword in self.sensitive_keywords):
            return LLMPrivacyClass.LOCAL_ONLY
        return LLMPrivacyClass.LOCAL_PREFERRED


class ComplexityEstimator:
    """Estimate routing complexity on a 1-5 scale."""

    def estimate(self, messages: list[LLMMessage], task_type: LLMTaskType) -> int:
        text = _joined_user_text(messages)
        length_score = 1 if len(text) < 300 else 2 if len(text) < 1200 else 3
        task_bonus = 1 if task_type in {LLMTaskType.CODING, LLMTaskType.PLANNING, LLMTaskType.REASONING} else 0
        return max(1, min(5, length_score + task_bonus))


def _joined_user_text(messages: list[LLMMessage]) -> str:
    parts = [message.content for message in messages if message.role == LLMRole.USER]
    if not parts:
        parts = [message.content for message in messages]
    return " ".join(parts).lower()
