from collections.abc import Iterator
import threading
from uuid import uuid4

from jarvis_brain.llm.consensus_engine import LLMConsensusEngine
from jarvis_brain.llm.intelligence_classifiers import CapabilityClassifier, ComplexityEstimator, PrivacyClassifier
from jarvis_brain.llm.llm_router_config import LLMRouterConfig
from jarvis_brain.llm.model_registry import LLMModelRegistry
from jarvis_brain.llm.omniroute.classifier import IntelligentTaskClassifier
from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.policy import OmniRouteSelectionPolicy
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import RouteClassification, RouteSelection
from jarvis_brain.llm.provider_registry import LLMProviderRegistry
from jarvis_brain.llm.router_telemetry import LLMRouterTelemetry, llm_router_telemetry
from jarvis_brain.llm.routing_observatory import LLMRoutingObservatory, llm_routing_observatory
from jarvis_platform.observability.event_logger import observability_event_logger
from jarvis_platform.observability.schemas import EventCategory
from jarvis_platform.schemas.llm import LLMProviderName, LLMRequest, LLMResponse, LLMStatus, LLMTaskType
from jarvis_platform.schemas.llm_router import (
    LLMPrivacyClass,
    LLMProviderHealth,
    LLMRouteDecisionStatus,
    LLMRouterStatistics,
    LLMRoutingDecision,
)


class IntelligenceRouter:
    """Single safe gateway for selecting LLM intelligence."""

    def __init__(
        self,
        config: LLMRouterConfig | None = None,
        model_registry: LLMModelRegistry | None = None,
        provider_registry: LLMProviderRegistry | None = None,
        telemetry: LLMRouterTelemetry | None = None,
        capability_classifier: CapabilityClassifier | None = None,
        privacy_classifier: PrivacyClassifier | None = None,
        complexity_estimator: ComplexityEstimator | None = None,
        consensus_engine: LLMConsensusEngine | None = None,
        intelligent_task_classifier: IntelligentTaskClassifier | None = None,
        omniroute_settings: OmniRouteSettings | None = None,
        omniroute_registry: OmniRouteRouteRegistry | None = None,
        omniroute_policy: OmniRouteSelectionPolicy | None = None,
        event_logger=None,
        routing_observatory: LLMRoutingObservatory | None = None,
    ) -> None:
        self.config = config or LLMRouterConfig()
        self.model_registry = model_registry or LLMModelRegistry()
        self.provider_registry = provider_registry or LLMProviderRegistry()
        self.telemetry = telemetry or llm_router_telemetry
        self.capability_classifier = capability_classifier or CapabilityClassifier()
        self.privacy_classifier = privacy_classifier or PrivacyClassifier()
        self.complexity_estimator = complexity_estimator or ComplexityEstimator()
        self.consensus_engine = consensus_engine or LLMConsensusEngine()
        provider_settings = getattr(self.provider_registry, "omniroute_settings", None)
        self.omniroute_settings = (
            omniroute_settings or provider_settings or OmniRouteSettings()
        )
        provider_routes = getattr(self.provider_registry, "omniroute_registry", None)
        self.omniroute_registry = (
            omniroute_registry
            or provider_routes
            or OmniRouteRouteRegistry(self.omniroute_settings)
        )
        self.intelligent_task_classifier = (
            intelligent_task_classifier or IntelligentTaskClassifier()
        )
        self.omniroute_policy = omniroute_policy or OmniRouteSelectionPolicy(
            self.omniroute_registry, self.omniroute_settings
        )
        self.event_logger = event_logger or observability_event_logger
        self.routing_observatory = routing_observatory or (
            LLMRoutingObservatory(self.event_logger)
            if event_logger is not None
            else llm_routing_observatory
        )
        self._registry_refresh_lock = threading.RLock()

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Route and generate with bounded retry/fallback behavior."""
        started_at = self.telemetry.start_timer()
        source = str(request.metadata.get("source") or "unknown")[:40]
        self.routing_observatory.request_received(request.request_id, source=source)
        task_type = self.capability_classifier.classify(request.messages, request.metadata)
        privacy_class = self.privacy_classifier.classify(request.messages, request.metadata)
        complexity = self.complexity_estimator.estimate(request.messages, task_type)
        classification = self.intelligent_task_classifier.classify(
            request.messages, request.metadata
        )
        self.routing_observatory.classified(
            request.request_id,
            task_category=classification.task_category.value,
            privacy_class=privacy_class.value,
            complexity=complexity,
            required_capabilities=[item.value for item in classification.required_capabilities],
        )
        gateway_selection = self.omniroute_policy.select(classification)
        if gateway_selection.rejected_routes:
            self.routing_observatory.blocked(
                request.request_id,
                task_category=classification.task_category.value,
                privacy_class=privacy_class.value,
                reason_codes=sorted(
                    {reason for reasons in gateway_selection.rejected_routes.values() for reason in reasons}
                ),
            )
        gateway_attempted = False
        failed_provider_id: str | None = None
        if gateway_selection.selected and gateway_selection.route is not None:
            gateway_attempted = True
            route = gateway_selection.route
            decision = self._decision(
                task_type=task_type,
                privacy_class=privacy_class,
                complexity=complexity,
                provider=LLMProviderName.OMNIROUTE,
                model=route.model_id,
                chain=[LLMProviderName.OMNIROUTE, LLMProviderName.OLLAMA],
                fallback_used=False,
                reason="Selected an explicitly approved OmniRoute route.",
                metadata=self._gateway_metadata(
                    classification, gateway_selection, request.request_id
                ),
            )
            provider = self.provider_registry.create_omniroute_provider(route)
            self.routing_observatory.route_selected(
                decision,
                request_id=request.request_id,
                source=source,
                gateway="omniroute",
                provider_id=route.provider_id,
                route_id=route.route_id,
                reason_codes=gateway_selection.reason_codes,
                rejected_candidates=gateway_selection.rejected_routes,
            )
            self.routing_observatory.connecting(request.request_id)
            self._log_gateway_event(
                "omniroute_route_selected",
                "Jarvis selected an approved gateway route.",
                decision,
            )
            if provider.is_available():
                routed_request = self._routed_request(
                    request,
                    provider.name,
                    route.model_id,
                    task_type,
                    privacy_class,
                    complexity,
                    decision,
                )
                response = self._try_provider(provider, routed_request)
                if response.status == LLMStatus.SUCCESS:
                    self.routing_observatory.validating(request.request_id)
                    self.routing_observatory.completed(request.request_id)
                    self._log_gateway_event(
                        "omniroute_generation_completed",
                        "The selected gateway route completed safely.",
                        decision,
                    )
                    self.telemetry.record_success(
                        provider.name, decision, started_at, fallback_used=False
                    )
                    return self._with_router_metadata(response, decision)
            failed_provider_id = route.provider_id
            self.routing_observatory.failed(
                request.request_id,
                safe_reason="PROVIDER_UNAVAILABLE",
                fallback_allowed=True,
            )
            self.telemetry.record_failure(LLMProviderName.OMNIROUTE, decision)
            self._log_gateway_event(
                "omniroute_fallback_started",
                "The gateway route failed before visible output; Jarvis is using its local fallback.",
                decision,
            )
        chain = self._fallback_chain(task_type, privacy_class, request.provider)

        last_decision: LLMRoutingDecision | None = None
        for index, provider_name in enumerate(chain):
            model = self.model_registry.select_model(provider_name, task_type)
            decision = self._decision(
                task_type=task_type,
                privacy_class=privacy_class,
                complexity=complexity,
                provider=provider_name,
                model=model,
                chain=chain[index:],
                fallback_used=gateway_attempted or index > 0,
                reason=self._reason(provider_name, task_type, privacy_class, gateway_attempted or index > 0),
            )
            last_decision = decision
            self.routing_observatory.route_selected(
                decision,
                request_id=request.request_id,
                source=source,
                gateway="direct",
                provider_id=provider_name.value,
                reason_codes=[
                    "LOCAL_FALLBACK" if gateway_attempted or index > 0 else "TASK_MATCH",
                    "PRIVACY_ALLOWED",
                ],
                fallback_from=failed_provider_id if gateway_attempted or index > 0 else None,
                fallback_reason="PRIMARY_FAILED_BEFORE_OUTPUT" if gateway_attempted or index > 0 else None,
            )
            provider = self.provider_registry.create_provider(provider_name, model)
            if provider is None:
                self.routing_observatory.failed(
                    request.request_id,
                    safe_reason="PROVIDER_NOT_CONFIGURED",
                    fallback_allowed=index < len(chain) - 1,
                )
                failed_provider_id = provider_name.value
                self.telemetry.record_failure(provider_name, decision)
                continue
            if not provider.is_available():
                self.routing_observatory.failed(
                    request.request_id,
                    safe_reason="PROVIDER_UNAVAILABLE",
                    fallback_allowed=index < len(chain) - 1,
                )
                failed_provider_id = provider_name.value
                self.telemetry.record_failure(provider.name, decision)
                continue
            self.routing_observatory.connecting(request.request_id)
            routed_request = self._routed_request(
                request,
                provider.name,
                model,
                task_type,
                privacy_class,
                complexity,
                decision,
            )
            response = self._try_provider(provider, routed_request)
            if response.status == LLMStatus.SUCCESS:
                self.routing_observatory.validating(request.request_id)
                self.routing_observatory.completed(request.request_id)
                self.telemetry.record_success(provider.name, decision, started_at, fallback_used=gateway_attempted or index > 0)
                return self._with_router_metadata(response, decision)
            self.routing_observatory.failed(
                request.request_id,
                safe_reason="PROVIDER_ERROR",
                fallback_allowed=index < len(chain) - 1,
            )
            failed_provider_id = provider_name.value
            self.telemetry.record_failure(provider.name, decision)

        safe_decision = last_decision or self._decision(
            task_type=task_type,
            privacy_class=privacy_class,
            complexity=complexity,
            provider=LLMProviderName.MOCK,
            model="mock-model",
            chain=[],
            fallback_used=True,
            reason="No configured provider was available.",
        )
        return LLMResponse(
            response_id=str(uuid4()),
            request_id=request.request_id,
            provider=safe_decision.selected_provider,
            model=safe_decision.selected_model,
            task_type=task_type,
            status=LLMStatus.ERROR,
            content="I could not reach an available language model safely.",
            error_message="No configured LLM provider was available.",
            raw_metadata={"router_decision": safe_decision.model_dump(mode="json"), "tools_allowed": False},
        )

    def route_only(self, messages: list, metadata: dict | None = None) -> LLMRoutingDecision:
        """Return a routing decision without calling a provider."""
        request = LLMRequest(
            request_id=str(uuid4()),
            provider=LLMProviderName.MOCK,
            model="mock-model",
            messages=messages,
            metadata=metadata or {},
        )
        task_type = self.capability_classifier.classify(request.messages, request.metadata)
        privacy = self.privacy_classifier.classify(request.messages, request.metadata)
        complexity = self.complexity_estimator.estimate(request.messages, task_type)
        classification = self.intelligent_task_classifier.classify(
            request.messages, request.metadata
        )
        gateway_selection = self.omniroute_policy.select(classification)
        if gateway_selection.selected and gateway_selection.route is not None:
            route = gateway_selection.route
            return self._decision(
                task_type,
                privacy,
                complexity,
                LLMProviderName.OMNIROUTE,
                route.model_id,
                [LLMProviderName.OMNIROUTE, LLMProviderName.OLLAMA],
                False,
                "Preview selected an explicitly approved OmniRoute route.",
                metadata=self._gateway_metadata(
                    classification, gateway_selection, request.request_id
                ),
            )
        chain = self._fallback_chain(task_type, privacy, request.provider)
        provider = chain[0] if chain else LLMProviderName.MOCK
        model = self.model_registry.select_model(provider, task_type)
        return self._decision(task_type, privacy, complexity, provider, model, chain, False, "Preview routing decision.")

    def generate_stream(self, request: LLMRequest) -> Iterator[dict]:
        """Stream from one approved route, falling back only before visible output."""
        started_at = self.telemetry.start_timer()
        source = str(request.metadata.get("source") or "unknown")[:40]
        self.routing_observatory.request_received(request.request_id, source=source)
        task_type = self.capability_classifier.classify(request.messages, request.metadata)
        privacy = self.privacy_classifier.classify(request.messages, request.metadata)
        complexity = self.complexity_estimator.estimate(request.messages, task_type)
        classification = self.intelligent_task_classifier.classify(
            request.messages, request.metadata
        )
        self.routing_observatory.classified(
            request.request_id,
            task_category=classification.task_category.value,
            privacy_class=privacy.value,
            complexity=complexity,
            required_capabilities=[item.value for item in classification.required_capabilities],
        )
        selection = self.omniroute_policy.select(classification)
        if selection.rejected_routes:
            self.routing_observatory.blocked(
                request.request_id,
                task_category=classification.task_category.value,
                privacy_class=privacy.value,
                reason_codes=sorted(
                    {reason for reasons in selection.rejected_routes.values() for reason in reasons}
                ),
            )
        candidates: list[tuple[LLMProviderName, str, object, dict]] = []
        if (
            self.omniroute_settings.streaming_enabled
            and selection.selected
            and selection.route is not None
            and selection.route.supports_streaming
        ):
            route = selection.route
            candidates.append(
                (
                    LLMProviderName.OMNIROUTE,
                    route.model_id,
                    self.provider_registry.create_omniroute_provider(route),
                    self._gateway_metadata(
                        classification, selection, request.request_id
                    ),
                )
            )
        for provider_name in self._fallback_chain(task_type, privacy, request.provider):
            model = self.model_registry.select_model(provider_name, task_type)
            provider = self.provider_registry.create_provider(provider_name, model)
            if provider is not None:
                candidates.append((provider_name, model, provider, {}))

        failed_provider_id: str | None = None
        for index, (provider_name, model, provider, gateway_metadata) in enumerate(candidates):
            if not provider.is_available():
                failed_provider_id = (
                    str(gateway_metadata.get("selected_provider_id") or provider_name.value)
                )
                continue
            decision = self._decision(
                task_type,
                privacy,
                complexity,
                provider_name,
                model,
                [item[0] for item in candidates[index:]],
                index > 0,
                self._reason(provider_name, task_type, privacy, index > 0),
                metadata=gateway_metadata,
            )
            routed = self._routed_request(
                request, provider_name, model, task_type, privacy, complexity, decision
            )
            provider_id = str(
                gateway_metadata.get("selected_provider_id") or provider_name.value
            )
            self.routing_observatory.route_selected(
                decision,
                request_id=request.request_id,
                source=source,
                gateway="omniroute" if provider_name == LLMProviderName.OMNIROUTE else "direct",
                provider_id=provider_id,
                route_id=(
                    str(gateway_metadata.get("selected_route_id"))
                    if provider_name == LLMProviderName.OMNIROUTE
                    and gateway_metadata.get("selected_route_id")
                    else None
                ),
                reason_codes=list(gateway_metadata.get("selection_reason_codes") or ["TASK_MATCH", "PRIVACY_ALLOWED"]),
                rejected_candidates=selection.rejected_routes if provider_name == LLMProviderName.OMNIROUTE else None,
                fallback_from=failed_provider_id if index > 0 else None,
                fallback_reason="PRIMARY_FAILED_BEFORE_OUTPUT" if index > 0 else None,
            )
            self.routing_observatory.connecting(request.request_id)
            if provider_name == LLMProviderName.OMNIROUTE:
                self._log_gateway_event(
                    "omniroute_stream_started",
                    "Jarvis opened a policy-approved gateway stream.",
                    decision,
                )
            visible_output = False
            chunk_index = 0
            stream = provider.generate_stream(routed)
            try:
                for event in stream:
                    kind = event.get("type")
                    if kind == "token" and event.get("content"):
                        visible_output = True
                        chunk_index += 1
                        self.routing_observatory.stream_delta(
                            request.request_id, chunk_index
                        )
                        yield event
                        continue
                    if kind == "done":
                        self.routing_observatory.validating(request.request_id)
                        metadata = dict(event.get("metadata") or {})
                        metadata.update(
                            {
                                "provider": provider_name.value,
                                "model": model,
                                "router_decision": decision.model_dump(mode="json"),
                                "fallback_used": index > 0,
                            }
                        )
                        yield {"type": "done", "metadata": metadata}
                        self.telemetry.record_success(
                            provider_name, decision, started_at, fallback_used=index > 0
                        )
                        self.routing_observatory.completed(request.request_id)
                        if provider_name == LLMProviderName.OMNIROUTE:
                            self._log_gateway_event(
                                "omniroute_stream_completed",
                                "The selected gateway stream completed safely.",
                                decision,
                            )
                        return
                    if kind == "error":
                        self.telemetry.record_failure(provider_name, decision)
                        self.routing_observatory.failed(
                            request.request_id,
                            safe_reason=(
                                "STREAM_FAILED_AFTER_OUTPUT"
                                if visible_output
                                else "STREAM_FAILED_BEFORE_OUTPUT"
                            ),
                            fallback_allowed=not visible_output,
                        )
                        failed_provider_id = provider_id
                        if visible_output:
                            yield {
                                "type": "error",
                                "message": "The selected model stream became unavailable.",
                                "metadata": {
                                    "provider": provider_name.value,
                                    "model": model,
                                    "fallback_blocked_after_visible_output": True,
                                },
                            }
                            return
                        if provider_name == LLMProviderName.OMNIROUTE:
                            self._log_gateway_event(
                                "omniroute_fallback_started",
                                "The gateway stream failed before visible output; Jarvis is using its local fallback.",
                                decision,
                            )
                        break
            finally:
                close = getattr(stream, "close", None)
                if callable(close):
                    close()
        yield {
            "type": "error",
            "message": "No configured LLM provider was available.",
            "metadata": {"fallback_exhausted": True},
        }

    def list_provider_health(self) -> list[LLMProviderHealth]:
        """Return safe public health for all known providers."""
        models_by_provider: dict[LLMProviderName, list[str]] = {}
        for model in self.model_registry.list_models():
            models_by_provider.setdefault(model.provider, []).append(model.model)
        return [
            self.provider_registry.health(provider, models_by_provider.get(provider, []))
            for provider in self.provider_registry.list_provider_names()
        ]

    def statistics(self) -> LLMRouterStatistics:
        return self.telemetry.snapshot()

    def refresh_omniroute_registry(self) -> bool:
        """Reload reviewed route configuration without changing routing authority."""
        with self._registry_refresh_lock:
            settings = OmniRouteSettings()
            registry = OmniRouteRouteRegistry(settings)
            before = [
                route.model_dump(mode="json")
                for route in self.omniroute_registry.list_routes()
            ]
            after = [route.model_dump(mode="json") for route in registry.list_routes()]
            settings_changed = (
                settings.routing_configuration()
                != self.omniroute_settings.routing_configuration()
            )
            if before == after and not settings_changed:
                return False
            self.omniroute_settings = settings
            self.omniroute_registry = registry
            self.omniroute_policy = OmniRouteSelectionPolicy(registry, settings)
            self.provider_registry.omniroute_settings = settings
            self.provider_registry.omniroute_registry = registry
            return True

    def _fallback_chain(self, task_type: LLMTaskType, privacy: LLMPrivacyClass, requested_provider: LLMProviderName) -> list[LLMProviderName]:
        if privacy == LLMPrivacyClass.LOCAL_ONLY:
            chain = [LLMProviderName.OLLAMA, LLMProviderName.MOCK]
        elif task_type == LLMTaskType.CODING:
            chain = self.config.coding_chain
        elif task_type in {LLMTaskType.REASONING, LLMTaskType.PLANNING, LLMTaskType.GUIDANCE}:
            chain = self.config.reasoning_chain
        else:
            chain = self.config.simple_chain
        if requested_provider not in {LLMProviderName.MOCK, LLMProviderName.UNKNOWN, LLMProviderName.OMNIROUTE} and requested_provider not in chain:
            chain = [requested_provider, *chain]
        return self._dedupe([provider for provider in chain if self._privacy_allows(provider, privacy)])

    def _privacy_allows(self, provider: LLMProviderName, privacy: LLMPrivacyClass) -> bool:
        if privacy == LLMPrivacyClass.LOCAL_ONLY:
            return provider in {LLMProviderName.OLLAMA, LLMProviderName.MOCK, LLMProviderName.LLAMA_CPP, LLMProviderName.LM_STUDIO, LLMProviderName.VLLM, LLMProviderName.AIRLLM}
        return True

    def _try_provider(self, provider, request: LLMRequest) -> LLMResponse:
        attempts = max(1, self.config.retry_count + 1)
        response: LLMResponse | None = None
        for _ in range(attempts):
            response = provider.generate(request)
            if response.status == LLMStatus.SUCCESS:
                return response
        return response or LLMResponse(
            response_id=str(uuid4()),
            request_id=request.request_id,
            provider=provider.name,
            model=request.model,
            task_type=request.task_type,
            status=LLMStatus.ERROR,
            content="",
            error_message="Provider did not return a response.",
        )

    def _decision(
        self,
        task_type: LLMTaskType,
        privacy_class: LLMPrivacyClass,
        complexity: int,
        provider: LLMProviderName,
        model: str,
        chain: list[LLMProviderName],
        fallback_used: bool,
        reason: str,
        metadata: dict | None = None,
    ) -> LLMRoutingDecision:
        return LLMRoutingDecision(
            decision_id=str(uuid4()),
            task_type=task_type,
            privacy_class=privacy_class,
            complexity_score=complexity,
            selected_provider=provider,
            selected_model=model,
            fallback_chain=chain,
            status=LLMRouteDecisionStatus.FALLBACK_USED if fallback_used else LLMRouteDecisionStatus.SELECTED,
            reason=reason,
            latency_budget_ms=self.config.latency_budget_ms,
            metadata=metadata or {},
        )

    def _routed_request(
        self,
        request: LLMRequest,
        provider: LLMProviderName,
        model: str,
        task_type: LLMTaskType,
        privacy_class: LLMPrivacyClass,
        complexity: int,
        decision: LLMRoutingDecision,
    ) -> LLMRequest:
        return request.model_copy(
            update={
                "provider": provider,
                "model": model,
                "task_type": task_type,
                "metadata": {
                    **request.metadata,
                    "router_decision": decision.model_dump(mode="json"),
                    "privacy_class": privacy_class.value,
                    "complexity_score": complexity,
                    "tools_allowed": False,
                },
            }
        )

    @staticmethod
    def _gateway_metadata(
        classification: RouteClassification,
        selection: RouteSelection,
        request_id: str,
    ) -> dict:
        route = selection.route
        if route is None:
            return {}
        return {
            "selected_gateway": "omniroute",
            "request_id": request_id,
            "selected_route_id": route.route_id,
            "selected_provider_id": route.provider_id,
            "route_locality": route.locality.value,
            "task_category": classification.task_category.value,
            "required_capabilities": [item.value for item in classification.required_capabilities],
            "selection_reason_codes": selection.reason_codes,
            "selection_score": selection.score,
            "tools_allowed": False,
        }

    def _reason(self, provider: LLMProviderName, task_type: LLMTaskType, privacy: LLMPrivacyClass, fallback_used: bool) -> str:
        prefix = "Fallback selected" if fallback_used else "Selected"
        return f"{prefix} {provider.value} for {task_type.value} with {privacy.value} privacy."

    def _with_router_metadata(self, response: LLMResponse, decision: LLMRoutingDecision) -> LLMResponse:
        metadata = dict(response.raw_metadata)
        metadata["router_decision"] = decision.model_dump(mode="json")
        metadata["tools_allowed"] = False
        return response.model_copy(update={"raw_metadata": metadata, "model": decision.selected_model, "task_type": decision.task_type})

    def _dedupe(self, values: list[LLMProviderName]) -> list[LLMProviderName]:
        seen: set[LLMProviderName] = set()
        result: list[LLMProviderName] = []
        for value in values:
            if value not in seen and value != LLMProviderName.UNKNOWN:
                seen.add(value)
                result.append(value)
        return result

    def _log_gateway_event(
        self, event_type: str, message: str, decision: LLMRoutingDecision
    ) -> None:
        metadata = decision.metadata
        self.event_logger.log_event(
            event_type,
            message,
            category=EventCategory.LLM,
            title="LLM gateway route",
            metadata={
                "request_id": metadata.get("request_id"),
                "decision_id": decision.decision_id,
                "task_category": metadata.get("task_category"),
                "privacy_class": decision.privacy_class.value,
                "selected_gateway": metadata.get("selected_gateway"),
                "selected_route_id": metadata.get("selected_route_id"),
                "selected_provider_id": metadata.get("selected_provider_id"),
                "selected_model_id": decision.selected_model,
                "route_locality": metadata.get("route_locality"),
                "selection_reason_codes": metadata.get("selection_reason_codes", []),
                "fallback_used": decision.status == LLMRouteDecisionStatus.FALLBACK_USED,
            },
            persist=False,
        )


def create_intelligence_router() -> IntelligenceRouter:
    return IntelligenceRouter()


intelligence_router = create_intelligence_router()
