from jarvis_brain.llm.intelligence_classifiers import CapabilityClassifier, ComplexityEstimator, PrivacyClassifier
from jarvis_brain.llm.omniroute.schemas import ModelCapability, RouteClassification, TaskCategory
from jarvis_platform.schemas.llm import LLMMessage, LLMTaskType


class IntelligentTaskClassifier:
    """Enrich existing deterministic classifiers without becoming a second router."""

    def __init__(self) -> None:
        self.capability = CapabilityClassifier()
        self.privacy = PrivacyClassifier()
        self.complexity = ComplexityEstimator()

    def classify(self, messages: list[LLMMessage], metadata: dict | None = None) -> RouteClassification:
        metadata = dict(metadata or {})
        task = self.capability.classify(messages, metadata)
        privacy = self.privacy.classify(messages, metadata).value
        if privacy in {"cloud_allowed", "cloud_required"} and metadata.get(
            "_cloud_authorized"
        ) is not True:
            privacy = "local_preferred"
        score = self.complexity.estimate(messages, task)
        category = self._category(task, messages, metadata)
        required = self._capabilities(category)
        if privacy == "local_only":
            required = list(dict.fromkeys([*required, ModelCapability.LOCAL_EXECUTION]))
        return RouteClassification(
            task_category=category,
            complexity="high" if score >= 4 else "medium" if score >= 2 else "low",
            privacy=privacy,
            required_capabilities=required,
            latency_priority="interactive" if category in {TaskCategory.SIMPLE_CONVERSATION, TaskCategory.VOICE_CONVERSATION} else "balanced",
            quality_priority="high" if score >= 4 else "balanced",
            estimated_context_tokens=max(1, sum(len(item.content) for item in messages) // 4),
            reason_codes=[f"task:{task.value}", f"complexity:{score}", f"privacy:{privacy}"],
        )

    def _category(self, task: LLMTaskType, messages: list[LLMMessage], metadata: dict) -> TaskCategory:
        if metadata.get("source") == "voice" or task == LLMTaskType.VOICE:
            return TaskCategory.VOICE_CONVERSATION
        text = " ".join(item.content.lower() for item in messages)
        if "long context" in text:
            return TaskCategory.LONG_CONTEXT
        if metadata.get("system_command"):
            return TaskCategory.SYSTEM_COMMAND
        if metadata.get("private_document"):
            return TaskCategory.PRIVATE_DOCUMENT
        if task == LLMTaskType.CODING:
            if "review" in text:
                return TaskCategory.CODE_REVIEW
            if any(word in text for word in ("traceback", "debug", "bug", "failure")):
                return TaskCategory.DEBUGGING
            return TaskCategory.CODING
        mapping = {
            LLMTaskType.CONVERSATION: TaskCategory.SIMPLE_CONVERSATION,
            LLMTaskType.GENERAL: TaskCategory.SIMPLE_CONVERSATION,
            LLMTaskType.SUMMARIZATION: TaskCategory.SUMMARIZATION,
            LLMTaskType.PLANNING: TaskCategory.SOFTWARE_ARCHITECTURE,
            LLMTaskType.REASONING: TaskCategory.COMPLEX_REASONING,
            LLMTaskType.STRUCTURED_EXTRACTION: TaskCategory.STRUCTURED_EXTRACTION,
            LLMTaskType.RETRIEVAL: TaskCategory.RESEARCH_SYNTHESIS,
            LLMTaskType.WORLD_INTELLIGENCE: TaskCategory.RESEARCH_SYNTHESIS,
            LLMTaskType.CREATIVE: TaskCategory.CREATIVE_WRITING,
            LLMTaskType.VISION: TaskCategory.VISION,
            LLMTaskType.TOOL_PLANNING: TaskCategory.TOOL_PLANNING,
            LLMTaskType.INTENT_RESOLUTION: TaskCategory.CLASSIFICATION,
        }
        return mapping.get(task, TaskCategory.UNKNOWN)

    @staticmethod
    def _capabilities(category: TaskCategory) -> list[ModelCapability]:
        mapping = {
            TaskCategory.SIMPLE_CONVERSATION: [ModelCapability.GENERAL_CHAT, ModelCapability.LOW_LATENCY],
            TaskCategory.VOICE_CONVERSATION: [ModelCapability.GENERAL_CHAT, ModelCapability.LOW_LATENCY],
            TaskCategory.CLASSIFICATION: [ModelCapability.LOW_LATENCY, ModelCapability.STRUCTURED_OUTPUT],
            TaskCategory.CODING: [ModelCapability.CODING],
            TaskCategory.CODE_REVIEW: [ModelCapability.CODING, ModelCapability.REASONING],
            TaskCategory.DEBUGGING: [ModelCapability.CODING, ModelCapability.REASONING],
            TaskCategory.SOFTWARE_ARCHITECTURE: [ModelCapability.REASONING, ModelCapability.CODING, ModelCapability.LONG_CONTEXT],
            TaskCategory.COMPLEX_REASONING: [ModelCapability.REASONING, ModelCapability.HIGH_RELIABILITY],
            TaskCategory.LONG_CONTEXT: [ModelCapability.LONG_CONTEXT],
            TaskCategory.STRUCTURED_EXTRACTION: [ModelCapability.STRUCTURED_OUTPUT],
            TaskCategory.RESEARCH_SYNTHESIS: [ModelCapability.REASONING, ModelCapability.LONG_CONTEXT],
            TaskCategory.MULTILINGUAL: [ModelCapability.MULTILINGUAL],
            TaskCategory.VISION: [ModelCapability.VISION],
            TaskCategory.TOOL_PLANNING: [ModelCapability.REASONING, ModelCapability.STRUCTURED_OUTPUT],
            TaskCategory.PRIVATE_DOCUMENT: [ModelCapability.LOCAL_EXECUTION, ModelCapability.LONG_CONTEXT],
        }
        return mapping.get(category, [ModelCapability.GENERAL_CHAT])
