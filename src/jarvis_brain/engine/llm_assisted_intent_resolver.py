import os

from jarvis_brain.engine.intent_resolver import IntentResolver
from jarvis_brain.engine.llm_intent_prompt_builder import LLMIntentPromptBuilder
from jarvis_brain.engine.llm_intent_validator import LLMIntentValidator
from jarvis_brain.llm.safe_llm_service import SafeLLMService
from jarvis_platform.schemas.intent_result import IntentResult


class LLMAssistedIntentResolver:
    """Resolve intents with deterministic rules first and an optional LLM fallback."""

    def __init__(
        self,
        rule_resolver: IntentResolver | None = None,
        safe_llm_service: SafeLLMService | None = None,
        prompt_builder: LLMIntentPromptBuilder | None = None,
        validator: LLMIntentValidator | None = None,
        llm_confidence_threshold: float = 0.55,
        rule_confidence_threshold: float = 0.85,
        enabled: bool = False,
    ) -> None:
        """Create a rules-first resolver."""
        self.rule_resolver = rule_resolver or IntentResolver()
        self.safe_llm_service = safe_llm_service or SafeLLMService()
        self.prompt_builder = prompt_builder or LLMIntentPromptBuilder()
        self.validator = validator or LLMIntentValidator()
        self.llm_confidence_threshold = llm_confidence_threshold
        self.rule_confidence_threshold = rule_confidence_threshold
        self.enabled = enabled

    def resolve(
        self,
        raw_input: str,
        context: dict | None = None,
    ) -> IntentResult:
        """Resolve input without ever executing tools."""
        rule_result = self.rule_resolver.resolve(raw_input)
        if (
            rule_result.intent_type != "unknown"
            and rule_result.confidence >= self.rule_confidence_threshold
        ):
            return rule_result
        if not self.enabled:
            return rule_result

        response = self.safe_llm_service.generate(
            self.prompt_builder.build_messages(raw_input),
            metadata={
                "task_type": "intent_resolution",
                "context": context or {},
            },
        )
        if not response.is_success():
            return rule_result

        candidate = self.validator.parse_candidate(raw_input, response.content)
        if candidate is None:
            return self._with_rejection(rule_result, "LLM response was not valid JSON.")
        if candidate.confidence < self.llm_confidence_threshold:
            return self._with_rejection(
                rule_result,
                "LLM confidence was below the configured threshold.",
            )

        validation = self.validator.validate(candidate)
        if not validation.is_accepted():
            return self._with_rejection(rule_result, validation.reason)

        intent_type = validation.sanitized_intent_type or "unknown"
        action = validation.sanitized_action
        return IntentResult(
            name=intent_type,
            intent_type=intent_type,
            action=action,
            target=validation.sanitized_target,
            raw_input=raw_input,
            confidence=candidate.confidence,
            summary=candidate.reasoning_summary,
            entities=candidate.entities,
            metadata={
                **candidate.metadata,
                **validation.metadata,
                "llm_assisted": True,
                "llm_model": response.model,
                "llm_confidence": candidate.confidence,
                "original_rule_intent": rule_result.intent_type,
                "tools_executed": False,
            },
            requires_plan=intent_type == "plan" and action == "create_plan",
            needs_clarification=validation.decision.value == "needs_clarification",
        )

    def _with_rejection(self, result: IntentResult, reason: str) -> IntentResult:
        """Return the rule result with safe LLM rejection metadata."""
        metadata = dict(result.metadata)
        metadata.update(
            {
                "llm_assisted": False,
                "llm_rejected": True,
                "llm_reject_reason": reason,
            }
        )
        return result.model_copy(update={"metadata": metadata})


def create_llm_assisted_intent_resolver(
    rule_resolver: IntentResolver | None = None,
    safe_llm_service: SafeLLMService | None = None,
) -> LLMAssistedIntentResolver:
    """Create an assisted resolver from environment configuration."""
    return LLMAssistedIntentResolver(
        rule_resolver=rule_resolver,
        safe_llm_service=safe_llm_service,
        enabled=_env_truthy(os.getenv("LLM_INTENT_ENABLED")),
        rule_confidence_threshold=_float_from_env(
            "LLM_INTENT_RULE_CONFIDENCE_THRESHOLD",
            0.85,
        ),
        llm_confidence_threshold=_float_from_env(
            "LLM_INTENT_CONFIDENCE_THRESHOLD",
            0.55,
        ),
    )


def _env_truthy(value: str | None) -> bool:
    """Return True for common truthy environment values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    """Read a confidence threshold safely from the environment."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(1.0, max(0.0, value))
