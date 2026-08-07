import os

from jarvis_brain.engine.llm_plan_prompt_builder import LLMPlanPromptBuilder
from jarvis_brain.engine.llm_plan_validator import LLMPlanValidator
from jarvis_brain.engine.planner import Planner
from jarvis_brain.llm.safe_llm_service import SafeLLMService
from jarvis_platform.schemas.plan import ExecutionPlan


class LLMAssistedPlanner:
    """Run the rule planner first and optionally request a validated LLM proposal."""

    def __init__(
        self,
        rule_planner: Planner | None = None,
        safe_llm_service: SafeLLMService | None = None,
        prompt_builder: LLMPlanPromptBuilder | None = None,
        validator: LLMPlanValidator | None = None,
        enabled: bool = False,
        rule_plan_min_steps: int = 1,
        llm_confidence_threshold: float = 0.55,
    ) -> None:
        """Create a non-executing, rules-first planner."""
        self.rule_planner = rule_planner or Planner()
        self.safe_llm_service = safe_llm_service or SafeLLMService()
        self.prompt_builder = prompt_builder or LLMPlanPromptBuilder()
        self.validator = validator or LLMPlanValidator()
        self.enabled = enabled
        self.rule_plan_min_steps = max(1, rule_plan_min_steps)
        self.llm_confidence_threshold = llm_confidence_threshold

    def create_plan(
        self,
        raw_input: str,
        context: dict | None = None,
    ) -> ExecutionPlan:
        """Return a rule plan or a validated LLM plan proposal."""
        rule_plan = self.rule_planner.create_plan(raw_input)
        if not self.enabled:
            return rule_plan

        response = self.safe_llm_service.generate(
            self.prompt_builder.build_messages(raw_input, context=context),
            metadata={"task_type": "planning"},
        )
        if not response.is_success():
            return self._with_rejection(rule_plan, "LLM planning was unavailable.")

        candidate = self.validator.parse_candidate(raw_input, response.content)
        if candidate is None:
            return self._with_rejection(rule_plan, "LLM plan was not valid JSON.")
        if candidate.confidence < self.llm_confidence_threshold:
            return self._with_rejection(
                rule_plan,
                "LLM plan confidence was below the configured threshold.",
            )

        validation = self.validator.validate(candidate)
        if not validation.is_accepted():
            return self._with_rejection(rule_plan, validation.reason)

        plan = self.validator.to_execution_plan(candidate)
        metadata = dict(plan.metadata)
        metadata.update(
            {
                "llm_assisted": True,
                "llm_assisted_plan": True,
                "llm_model": response.model,
                "llm_confidence": candidate.confidence,
                "original_rule_plan_available": (
                    len(rule_plan.steps) >= self.rule_plan_min_steps
                ),
                "tools_executed": False,
            }
        )
        return plan.model_copy(update={"metadata": metadata})

    def _with_rejection(
        self,
        rule_plan: ExecutionPlan,
        reason: str,
    ) -> ExecutionPlan:
        """Return the rule plan with safe fallback metadata."""
        metadata = dict(rule_plan.metadata)
        metadata.update(
            {
                "llm_assisted": False,
                "llm_plan_rejected": True,
                "llm_reject_reason": reason,
            }
        )
        return rule_plan.model_copy(update={"metadata": metadata})


def create_llm_assisted_planner(
    rule_planner: Planner | None = None,
    safe_llm_service: SafeLLMService | None = None,
) -> LLMAssistedPlanner:
    """Create an assisted planner from environment configuration."""
    return LLMAssistedPlanner(
        rule_planner=rule_planner,
        safe_llm_service=safe_llm_service,
        enabled=_env_truthy(os.getenv("LLM_PLANNER_ENABLED")),
        llm_confidence_threshold=_float_from_env(
            "LLM_PLANNER_CONFIDENCE_THRESHOLD",
            0.55,
        ),
    )


def _env_truthy(value: str | None) -> bool:
    """Return True for common truthy environment values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _float_from_env(name: str, default: float) -> float:
    """Read a bounded confidence threshold from the environment."""
    try:
        value = float(os.getenv(name, str(default)))
    except ValueError:
        return default
    return min(1.0, max(0.0, value))
