from typing import Any
from uuid import uuid4

from jarvis_brain.orchestrator.agent_team_builder import AgentTeamBuilder
from jarvis_brain.orchestrator.capability_registry import BrainCapabilityRegistry
from jarvis_brain.orchestrator.context_builder import BrainContextBuilder
from jarvis_brain.orchestrator.execution_graph import ExecutionGraphBuilder
from jarvis_brain.orchestrator.prompt_builder import BrainOrchestratorPromptBuilder
from jarvis_brain.orchestrator.proposal_parser import BrainProposalParser
from jarvis_brain.orchestrator.verification import BrainOrchestrationValidator
from jarvis_brain.engine.intent_resolver import IntentResolver
from jarvis_brain.llm.safe_llm_service import SafeLLMService
from jarvis_platform.schemas.brain_orchestration import (
    BrainEventType,
    BrainExecutionNodeType,
    BrainIntelligenceMode,
    BrainIntentCandidate,
    BrainIntentType,
    BrainOrchestrationEvent,
    BrainOrchestrationResult,
    BrainOrchestratorProposal,
    BrainPlanStep,
    BrainProviderRequirement,
    BrainValidationStatus,
)
from jarvis_platform.schemas.intent_result import IntentResult
from jarvis_platform.schemas.llm import LLMStatus
from jarvis_platform.security.input_security_gateway import InputSecurityGateway


class BrainOrchestrator:
    """LLM-first orchestration foundation for Jarvis Brain v2.

    The orchestrator proposes understanding, planning, agents, and provider
    requirements. It never executes tools. BrainEngine and the deterministic
    safety kernel remain authoritative.
    """

    def __init__(
        self,
        safe_llm_service: SafeLLMService | None = None,
        context_builder: BrainContextBuilder | None = None,
        prompt_builder: BrainOrchestratorPromptBuilder | None = None,
        parser: BrainProposalParser | None = None,
        validator: BrainOrchestrationValidator | None = None,
        capability_registry: BrainCapabilityRegistry | None = None,
        agent_team_builder: AgentTeamBuilder | None = None,
        graph_builder: ExecutionGraphBuilder | None = None,
        fallback_resolver: IntentResolver | None = None,
        input_security_gateway: InputSecurityGateway | None = None,
        enabled: bool = True,
    ) -> None:
        self.safe_llm_service = safe_llm_service or SafeLLMService()
        self.context_builder = context_builder or BrainContextBuilder()
        self.prompt_builder = prompt_builder or BrainOrchestratorPromptBuilder()
        self.parser = parser or BrainProposalParser()
        self.validator = validator or BrainOrchestrationValidator()
        self.capability_registry = capability_registry or BrainCapabilityRegistry()
        self.agent_team_builder = agent_team_builder or AgentTeamBuilder()
        self.graph_builder = graph_builder or ExecutionGraphBuilder()
        self.fallback_resolver = fallback_resolver or IntentResolver()
        self.input_security_gateway = input_security_gateway or InputSecurityGateway()
        self.enabled = enabled

    def orchestrate(
        self,
        raw_input: str,
        request_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> BrainOrchestrationResult:
        """Run the LLM-first understanding/planning path with safe fallback."""
        request_id = request_id or str(uuid4())
        metadata = dict(metadata or {})
        events = [
            self._event(request_id, BrainEventType.BRAIN_STARTED, "Brain orchestration started.")
        ]
        security_report = self.input_security_gateway.inspect_input(
            source="brain_orchestrator",
            content=raw_input,
        )
        if security_report.get("is_suspicious") or self._looks_like_prompt_injection(raw_input):
            events.append(self._event(request_id, BrainEventType.FALLBACK_USED, "Input security gateway required deterministic fallback."))
            return self._fallback(
                raw_input,
                request_id,
                events,
                "Input security gateway marked the request suspicious.",
                metadata,
            )
        if not self.enabled:
            events.append(self._event(request_id, BrainEventType.FALLBACK_USED, "Brain Orchestrator disabled."))
            return self._fallback(raw_input, request_id, events, "Brain Orchestrator disabled.", metadata)

        context = self.context_builder.build(raw_input, metadata)
        events.append(self._event(request_id, BrainEventType.CONTEXT_BUILT, "Orchestration context assembled."))
        messages = self.prompt_builder.build_messages(raw_input, context)
        response = self.safe_llm_service.generate(
            messages=messages,
            metadata={"task_type": "structured_extraction", "brain_orchestration": True},
        )
        provider_metadata = self._provider_metadata(response.raw_metadata)
        if response.status != LLMStatus.SUCCESS:
            events.append(self._event(request_id, BrainEventType.FALLBACK_USED, "LLM orchestration provider unavailable."))
            return self._fallback(
                raw_input,
                request_id,
                events,
                response.error_message or "LLM orchestration provider unavailable.",
                {**metadata, **provider_metadata},
            )
        proposal = self.parser.parse(response.content)
        if proposal is None:
            events.append(self._event(request_id, BrainEventType.FALLBACK_USED, "Malformed LLM orchestration JSON."))
            return self._fallback(raw_input, request_id, events, "Malformed LLM orchestration JSON.", {**metadata, **provider_metadata})
        proposal = self._complete_proposal(proposal)
        accepted, reason = self.validator.validate(proposal)
        if not accepted:
            events.append(self._event(request_id, BrainEventType.FALLBACK_USED, reason))
            return self._fallback(raw_input, request_id, events, reason, {**metadata, **provider_metadata})

        graph = self.graph_builder.build(proposal)
        events.extend(
            [
                self._event(request_id, BrainEventType.INTENT_DETECTED, "LLM-first structured intent accepted."),
                self._event(request_id, BrainEventType.PLAN_BUILT, "LLM-first structured plan accepted."),
                self._event(request_id, BrainEventType.AGENT_TEAM_PROPOSED, "Agent team proposal validated."),
                self._event(request_id, BrainEventType.PROVIDER_SELECTED, "Provider requirements and router metadata recorded.", provider_metadata),
                self._event(request_id, BrainEventType.VERIFICATION_COMPLETED, "Orchestration proposal passed deterministic validation."),
            ]
        )
        return BrainOrchestrationResult(
            request_id=request_id,
            raw_input=raw_input,
            intelligence_mode=BrainIntelligenceMode.LLM_PRIMARY,
            validation_status=BrainValidationStatus.ACCEPTED,
            proposal=proposal,
            execution_graph=graph,
            events=events,
            selected_provider=provider_metadata.get("selected_provider"),
            selected_model=provider_metadata.get("selected_model"),
            confidence=proposal.confidence,
            metadata={**metadata, **provider_metadata},
        )

    def to_intent_result(self, result: BrainOrchestrationResult) -> IntentResult | None:
        """Convert an accepted orchestration intent to the existing IntentResult."""
        intent = result.primary_intent()
        if intent is None or result.validation_status != BrainValidationStatus.ACCEPTED:
            return None
        intent_type = self._legacy_intent_type(intent.intent_type)
        metadata = {
            "brain_orchestrated": True,
            "intelligence_mode": result.intelligence_mode.value,
            "validation_status": result.validation_status.value,
            "selected_provider": result.selected_provider,
            "selected_model": result.selected_model,
            **result.metadata,
        }
        return IntentResult(
            name=intent_type,
            intent_type=intent_type,
            action=intent.action,
            target=intent.target,
            goal=intent.goal,
            raw_input=result.raw_input,
            confidence=intent.confidence,
            entities=intent.entities,
            metadata=metadata,
            requires_plan=bool(result.proposal and result.proposal.plan_steps),
            needs_clarification=intent.needs_clarification,
        )

    def _complete_proposal(self, proposal: BrainOrchestratorProposal) -> BrainOrchestratorProposal:
        intent_types = [intent.intent_type for intent in proposal.intents] or [BrainIntentType.UNKNOWN]
        requirements = proposal.provider_requirements
        capabilities = requirements.capabilities or self.capability_registry.capabilities_for_intents(intent_types)
        agent_team = proposal.agent_team
        if not agent_team.roles:
            agent_team = self.agent_team_builder.propose_from_intents(intent_types)
        else:
            agent_team = self.agent_team_builder.validate(agent_team)
        if not proposal.plan_steps:
            proposal = proposal.model_copy(update={"plan_steps": [self._default_plan_step(intent_types[0])]})
        return proposal.model_copy(
            update={
                "agent_team": agent_team,
                "provider_requirements": requirements.model_copy(update={"capabilities": capabilities}),
            }
        )

    def _default_plan_step(self, intent_type: BrainIntentType) -> BrainPlanStep:
        return BrainPlanStep(
            step_id="step-1",
            order=1,
            title="Prepare safe response",
            description=f"Respond to the {intent_type.value} request after deterministic safety checks.",
            node_type=BrainExecutionNodeType.RESPONSE,
        )

    def _fallback(
        self,
        raw_input: str,
        request_id: str,
        events: list[BrainOrchestrationEvent],
        reason: str,
        metadata: dict[str, Any],
    ) -> BrainOrchestrationResult:
        fallback_intent = self.fallback_resolver.resolve(raw_input)
        proposal = BrainOrchestratorProposal(
            summary="Deterministic fallback intent was used.",
            intents=[
                BrainIntentCandidate(
                    intent_type=self._brain_intent_type(fallback_intent.intent_type),
                    action=fallback_intent.action,
                    target=fallback_intent.target,
                    goal=fallback_intent.goal,
                    confidence=fallback_intent.confidence,
                    entities=fallback_intent.entities,
                    needs_clarification=fallback_intent.needs_clarification,
                )
            ],
            confidence=fallback_intent.confidence,
            metadata={"legacy_intent_type": fallback_intent.intent_type},
        )
        return BrainOrchestrationResult(
            request_id=request_id,
            raw_input=raw_input,
            intelligence_mode=BrainIntelligenceMode.DETERMINISTIC_FALLBACK,
            validation_status=BrainValidationStatus.FALLBACK_USED,
            proposal=proposal,
            events=events,
            fallback_reason=reason,
            confidence=fallback_intent.confidence,
            metadata={**metadata, "fallback_reason": reason},
        )

    def _provider_metadata(self, metadata: dict[str, Any]) -> dict[str, Any]:
        decision = metadata.get("router_decision")
        if isinstance(decision, dict):
            return {
                "selected_provider": decision.get("selected_provider"),
                "selected_model": decision.get("selected_model"),
                "fallback_chain": decision.get("fallback_chain", []),
                "privacy_class": decision.get("privacy_class"),
                "routing_status": decision.get("status"),
            }
        return {
            "selected_provider": metadata.get("provider"),
            "selected_model": metadata.get("selected_model"),
        }

    def _event(
        self,
        request_id: str,
        event_type: BrainEventType,
        message: str,
        metadata: dict[str, Any] | None = None,
    ) -> BrainOrchestrationEvent:
        return BrainOrchestrationEvent(
            event_id=str(uuid4()),
            request_id=request_id,
            event_type=event_type,
            message=message,
            metadata=metadata or {},
        )

    def _brain_intent_type(self, legacy_intent_type: str) -> BrainIntentType:
        mapping = {
            "world_intelligence": BrainIntentType.WORLD,
            "coding_help": BrainIntentType.CODING,
            "universal_knowledge": BrainIntentType.RESEARCH,
            "plan": BrainIntentType.PLANNING,
            "goal": BrainIntentType.PLANNING,
            "approval_response": BrainIntentType.APPROVAL,
            "approval_confirm": BrainIntentType.APPROVAL,
            "approval_reject": BrainIntentType.APPROVAL,
            "calendar": BrainIntentType.CALENDAR,
            "task": BrainIntentType.AUTOMATION,
            "voice": BrainIntentType.VOICE,
            "vision": BrainIntentType.VISION,
            "general_question": BrainIntentType.CONVERSATION,
        }
        return mapping.get(legacy_intent_type, BrainIntentType.UNKNOWN)

    def _legacy_intent_type(self, brain_intent_type: BrainIntentType) -> str:
        mapping = {
            BrainIntentType.WORLD: "world_intelligence",
            BrainIntentType.CODING: "coding_help",
            BrainIntentType.RESEARCH: "universal_knowledge",
            BrainIntentType.PLANNING: "plan",
            BrainIntentType.APPROVAL: "approval_response",
            BrainIntentType.CALENDAR: "calendar",
            BrainIntentType.VOICE: "voice",
            BrainIntentType.VISION: "vision",
            BrainIntentType.CONVERSATION: "general_question",
            BrainIntentType.CHAT: "general_question",
            BrainIntentType.SYSTEM: "system",
            BrainIntentType.EXECUTION: "action",
            BrainIntentType.AUTOMATION: "task",
        }
        return mapping.get(brain_intent_type, brain_intent_type.value)

    def _looks_like_prompt_injection(self, raw_input: str) -> bool:
        text = raw_input.lower()
        phrases = (
            "ignore previous instructions",
            "bypass safety",
            "bypass security",
            "reveal hidden prompt",
            "show system prompt",
            "disable guardrails",
        )
        return any(phrase in text for phrase in phrases)
