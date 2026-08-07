from __future__ import annotations

from collections import deque
from datetime import datetime
import threading
from typing import TYPE_CHECKING

from jarvis_brain.llm.omniroute.schemas import OmniRouteRoute, RouteLocality
from jarvis_platform.observability.event_logger import EventLogger, observability_event_logger
from jarvis_platform.observability.schemas import EventCategory, EventSeverity
from jarvis_platform.schemas.common import utc_now
from jarvis_platform.schemas.llm import LLMProviderName
from jarvis_platform.schemas.llm_router import LLMRoutingDecision
from jarvis_brain.llm.topology_identity import (
    model_topology_id,
    provider_topology_id,
    route_topology_id,
)
from jarvis_platform.schemas.llm_topology import (
    ActiveLLMRoute,
    ModelTopologyEntry,
    ProviderCatalogDiff,
    ProviderTopologyNode,
    ProviderTopologyState,
    RouteLifecycleState,
    RouterTopologySnapshot,
    RoutingHistoryEntry,
)

if TYPE_CHECKING:
    from jarvis_brain.llm.intelligence_router import IntelligenceRouter


_DISPLAY_NAMES = {
    "ollama": "Ollama",
    "nvidia": "NVIDIA",
    "openai": "OpenAI",
    "anthropic": "Anthropic",
    "google": "Google",
    "gemini": "Gemini",
    "groq": "Groq",
    "openrouter": "OpenRouter",
    "together": "Together",
    "local": "Local Provider",
}


class LLMRoutingObservatory:
    """Bounded, observational projection of the authoritative LLM router."""

    def __init__(
        self,
        event_logger: EventLogger | None = None,
        *,
        history_limit: int = 50,
    ) -> None:
        self.event_logger = event_logger or observability_event_logger
        self.history_limit = max(1, min(history_limit, 50))
        self._routes: dict[str, ActiveLLMRoute] = {}
        self._history: deque[RoutingHistoryEntry] = deque(maxlen=self.history_limit)
        self._catalog: dict[str, ProviderTopologyNode] = {}
        self._lock = threading.RLock()

    def snapshot(self, router: IntelligenceRouter) -> RouterTopologySnapshot:
        providers = self._build_catalog(router)
        diff = self._reconcile_catalog(providers)
        with self._lock:
            active = self._current_route()
            history = [item.model_copy(deep=True) for item in reversed(self._history)]
        selected_id = active.topology_node_id if active else None
        selected_model = active.model_id if active else None
        selected_model_topology_id = active.topology_model_id if active else None
        providers = [
            provider.model_copy(
                update={
                    "selected": provider.topology_node_id == selected_id,
                    "active_model_id": (
                        selected_model if provider.topology_node_id == selected_id else None
                    ),
                    "active_model_topology_id": (
                        selected_model_topology_id
                        if provider.topology_node_id == selected_id
                        else None
                    ),
                    "streaming": bool(
                        provider.topology_node_id == selected_id
                        and active
                        and active.state == RouteLifecycleState.STREAMING
                    ),
                    "health": self._active_health(provider, active),
                }
            )
            for provider in providers
        ]
        return RouterTopologySnapshot(
            router_status="ready" if router.config.enabled else "disabled",
            active_route=active,
            providers=providers[:24],
            recent_route_changes=history,
            catalog_diff=diff,
        )

    def request_received(self, request_id: str, *, source: str) -> None:
        self._publish(
            "llm.request.received",
            "Jarvis received a model request.",
            {"request_id": request_id, "source": source},
        )

    def classified(
        self,
        request_id: str,
        *,
        task_category: str,
        privacy_class: str,
        complexity: int,
        required_capabilities: list[str],
    ) -> None:
        self._publish(
            "llm.request.classified",
            "Jarvis classified the request for safe model selection.",
            {
                "request_id": request_id,
                "task_category": task_category,
                "privacy_class": privacy_class,
                "complexity": complexity,
                "required_capabilities": required_capabilities[:12],
            },
        )

    def route_selected(
        self,
        decision: LLMRoutingDecision,
        *,
        request_id: str,
        source: str,
        gateway: str,
        provider_id: str,
        route_id: str | None = None,
        reason_codes: list[str] | None = None,
        rejected_candidates: dict[str, list[str]] | None = None,
        fallback_from: str | None = None,
        fallback_reason: str | None = None,
    ) -> None:
        with self._lock:
            previous = self._current_route()
        topology_node_id = provider_topology_id(
            gateway=gateway, provider_id=provider_id
        )
        topology_model_id = model_topology_id(
            gateway=gateway,
            provider_id=provider_id,
            model_id=decision.selected_model,
        )
        route = ActiveLLMRoute(
            request_id=request_id,
            source=source[:40] or "unknown",
            task_category=decision.task_type.value,
            complexity=decision.complexity_score,
            privacy_class=decision.privacy_class.value,
            gateway=gateway,
            provider_id=provider_id,
            model_id=decision.selected_model,
            route_id=route_id,
            topology_node_id=topology_node_id,
            topology_model_id=topology_model_id,
            topology_route_id=route_topology_id(
                gateway=gateway,
                provider_id=provider_id,
                model_id=decision.selected_model,
                route_id=route_id,
            ),
            state=RouteLifecycleState.SELECTED,
            selection_reason_codes=(reason_codes or [])[:12],
            rejected_candidates={
                key: value[:8]
                for key, value in list((rejected_candidates or {}).items())[:12]
            },
            fallback_from=fallback_from,
            fallback_reason=fallback_reason,
        )
        with self._lock:
            self._routes[request_id] = route
            if len(self._routes) > 64:
                oldest = min(self._routes.values(), key=lambda item: item.started_at)
                self._routes.pop(oldest.request_id, None)
        self._publish_route("llm.route.selected", "Jarvis selected a model route.", route)
        if previous and previous.topology_node_id != route.topology_node_id:
            self._publish_route(
                "llm.route.changed",
                "Jarvis changed the selected provider route.",
                route,
                extra={
                    "previous_provider_id": previous.provider_id,
                    "previous_topology_node_id": previous.topology_node_id,
                },
            )
        if fallback_from:
            self._publish_route(
                "llm.route.fallback",
                "Jarvis selected a fallback before visible output began.",
                route,
            )

    def connecting(self, request_id: str) -> None:
        self._transition(
            request_id,
            RouteLifecycleState.CONNECTING,
            "llm.route.connecting",
            "Jarvis is connecting to the selected provider.",
        )

    def stream_delta(self, request_id: str, chunk_index: int) -> None:
        with self._lock:
            route = self._matching(request_id)
            if route is None:
                return
            now = utc_now()
            first_token = route.first_token_at or now
            elapsed = (now - route.started_at).total_seconds() * 1000
            self._routes[request_id] = route.model_copy(
                update={
                    "state": RouteLifecycleState.STREAMING,
                    "first_token_at": first_token,
                    "time_to_first_token_ms": round(
                        (first_token - route.started_at).total_seconds() * 1000, 3
                    ),
                    "chunk_count": chunk_index,
                    "visible_output": True,
                }
            )
            current = self._routes[request_id]
        if chunk_index == 1:
            self._publish_route(
                "llm.stream.started",
                "The selected model began streaming a safe response.",
                current,
            )
        if chunk_index == 1 or chunk_index % 8 == 0:
            self._publish(
                "llm.stream.delta",
                "A safe response stream is active.",
                {
                    "request_id": request_id,
                    "provider_id": current.provider_id,
                    "model_id": current.model_id,
                    "topology_node_id": current.topology_node_id,
                    "topology_model_id": current.topology_model_id,
                    "chunk_index": chunk_index,
                    "elapsed_ms": round(elapsed, 3),
                },
            )

    def validating(self, request_id: str) -> None:
        self._transition(
            request_id,
            RouteLifecycleState.VALIDATING,
            "llm.stream.completed",
            "Model streaming completed and output validation is active.",
        )

    def completed(self, request_id: str) -> None:
        self._terminal(
            request_id,
            RouteLifecycleState.COMPLETED,
            "llm.route.completed",
            "The selected model route completed.",
        )

    def failed(
        self,
        request_id: str,
        *,
        safe_reason: str,
        fallback_allowed: bool,
    ) -> None:
        self._terminal(
            request_id,
            RouteLifecycleState.FAILED,
            "llm.route.failed",
            "The selected model route failed safely.",
            safe_reason=safe_reason,
            severity=EventSeverity.WARNING,
            extra={"fallback_allowed": fallback_allowed},
        )

    def blocked(
        self,
        request_id: str,
        *,
        task_category: str,
        privacy_class: str,
        reason_codes: list[str],
    ) -> None:
        self._publish(
            "llm.route.blocked",
            "A candidate route was blocked by Jarvis policy.",
            {
                "request_id": request_id,
                "task_category": task_category,
                "privacy_class": privacy_class,
                "reason_codes": reason_codes[:12],
            },
        )

    def _transition(
        self,
        request_id: str,
        state: RouteLifecycleState,
        event_type: str,
        message: str,
    ) -> None:
        with self._lock:
            route = self._matching(request_id)
            if route is None:
                return
            self._routes[request_id] = route.model_copy(update={"state": state})
            current = self._routes[request_id]
        self._publish_route(event_type, message, current)

    def _terminal(
        self,
        request_id: str,
        state: RouteLifecycleState,
        event_type: str,
        message: str,
        *,
        safe_reason: str | None = None,
        severity: EventSeverity = EventSeverity.INFO,
        extra: dict | None = None,
    ) -> None:
        with self._lock:
            route = self._matching(request_id)
            if route is None:
                return
            completed_at = utc_now()
            total_latency = round(
                (completed_at - route.started_at).total_seconds() * 1000, 3
            )
            route = route.model_copy(
                update={
                    "state": state,
                    "completed_at": completed_at,
                    "total_latency_ms": total_latency,
                }
            )
            self._routes[request_id] = route
            self._history.append(
                RoutingHistoryEntry(
                    request_id=route.request_id,
                    timestamp=completed_at,
                    source=route.source,
                    task_category=route.task_category,
                    privacy_class=route.privacy_class,
                    gateway=route.gateway,
                    provider_id=route.provider_id,
                    model_id=route.model_id,
                    route_id=route.route_id,
                    topology_node_id=route.topology_node_id,
                    topology_model_id=route.topology_model_id,
                    topology_route_id=route.topology_route_id,
                    state=state,
                    fallback_used=bool(route.fallback_from),
                    policy_blocked=state == RouteLifecycleState.BLOCKED,
                    total_latency_ms=total_latency,
                    safe_reason=safe_reason,
                )
            )
        self._publish_route(event_type, message, route, severity=severity, extra=extra)

    def _matching(self, request_id: str) -> ActiveLLMRoute | None:
        return self._routes.get(request_id)

    def _current_route(self) -> ActiveLLMRoute | None:
        active_states = {
            RouteLifecycleState.SELECTED,
            RouteLifecycleState.CONNECTING,
            RouteLifecycleState.STREAMING,
            RouteLifecycleState.VALIDATING,
        }
        routes = list(self._routes.values())
        candidates = [route for route in routes if route.state in active_states]
        selected = max(candidates or routes, key=lambda item: item.started_at, default=None)
        return selected.model_copy(deep=True) if selected else None

    def _build_catalog(self, router: IntelligenceRouter) -> list[ProviderTopologyNode]:
        health_by_id = {
            item.provider.value: item for item in router.list_provider_health()
        }
        direct_models = [
            item
            for item in router.model_registry.list_models()
            if item.provider == LLMProviderName.OLLAMA
        ]
        ollama_health = health_by_id.get("ollama")
        direct_ollama_id = provider_topology_id(
            gateway="direct", provider_id="ollama"
        )
        nodes: dict[str, ProviderTopologyNode] = {
            direct_ollama_id: ProviderTopologyNode(
                topology_node_id=direct_ollama_id,
                provider_id="ollama",
                display_name="Direct Ollama",
                icon_key="ollama",
                configured=bool(direct_models),
                approved=True,
                available=bool(ollama_health and ollama_health.available),
                health=(
                    ProviderTopologyState.AVAILABLE
                    if ollama_health and ollama_health.available
                    else ProviderTopologyState.UNAVAILABLE
                ),
                locality="local",
                gateway="direct",
                policy_status="allowed",
                model_count=len(direct_models),
                approved_model_count=len(direct_models),
                latency_ms=ollama_health.average_latency_ms if ollama_health else None,
                terms_status="local_runtime",
                models=[
                    ModelTopologyEntry(
                        topology_model_id=model_topology_id(
                            gateway="direct",
                            provider_id="ollama",
                            model_id=item.model,
                        ),
                        topology_route_id=route_topology_id(
                            gateway="direct",
                            provider_id="ollama",
                            model_id=item.model,
                            route_id=None,
                        ),
                        provider_id="ollama",
                        gateway="direct",
                        model_id=item.model,
                        display_name=item.model,
                        approved=item.enabled,
                        available=bool(ollama_health and ollama_health.available),
                        capabilities=self._direct_capabilities(item),
                        task_categories=[task.value for task in item.task_types],
                        context_window=item.context_length,
                        locality="local",
                        privacy_class="local_only",
                        streaming=item.supports_streaming,
                        tools=item.supports_tool_calling,
                        structured_output=item.supports_structured_output,
                        terms_status="local_runtime",
                    )
                    for item in direct_models
                ],
            )
        }
        for route in router.omniroute_registry.list_routes():
            topology_node_id = provider_topology_id(
                gateway=route.gateway, provider_id=route.provider_id
            )
            node = nodes.get(topology_node_id)
            model = self._gateway_model(router, route)
            if node is None:
                policy_status, health = self._route_status(router, route)
                node = ProviderTopologyNode(
                    topology_node_id=topology_node_id,
                    provider_id=route.provider_id,
                    display_name="OmniRoute · "
                    + _DISPLAY_NAMES.get(
                        route.provider_id.lower(),
                        route.provider_id.replace("-", " ").title(),
                    ),
                    icon_key=route.provider_id.lower(),
                    configured=True,
                    approved=router.omniroute_registry.is_operator_approved(route),
                    available=health == ProviderTopologyState.AVAILABLE,
                    health=health,
                    locality=route.locality.value,
                    gateway="omniroute",
                    development_only=route.development_only,
                    policy_status=policy_status,
                    quota_status=route.quota_class,
                    terms_status=route.terms_status,
                    models=[],
                )
            node.models.append(model)
            node.model_count = len(node.models)
            node.approved_model_count = sum(1 for item in node.models if item.approved)
            node.approved = node.approved or model.approved
            node.development_only = node.development_only or route.development_only
            nodes[topology_node_id] = node
        return [nodes[key] for key in sorted(nodes)]

    @staticmethod
    def _direct_capabilities(item) -> list[str]:
        capabilities = ["general_chat"]
        if item.supports_streaming:
            capabilities.append("streaming")
        if item.supports_structured_output:
            capabilities.append("structured_output")
        if item.supports_vision:
            capabilities.append("vision")
        if item.supports_tool_calling:
            capabilities.append("tool_calling")
        return capabilities

    @staticmethod
    def _gateway_model(router: IntelligenceRouter, route: OmniRouteRoute) -> ModelTopologyEntry:
        approved = router.omniroute_registry.is_operator_approved(route)
        reason = None
        if not route.enabled:
            reason = "ROUTE_DISABLED"
        elif route.terms_status != "approved":
            reason = "TERMS_NOT_APPROVED"
        elif route.locality == RouteLocality.CLOUD and not router.omniroute_settings.allow_cloud:
            reason = "CLOUD_DISABLED"
        elif not router.omniroute_settings.api_key:
            reason = "CREDENTIAL_MISSING"
        return ModelTopologyEntry(
            topology_model_id=model_topology_id(
                gateway=route.gateway,
                provider_id=route.provider_id,
                model_id=route.model_id,
            ),
            topology_route_id=route_topology_id(
                gateway=route.gateway,
                provider_id=route.provider_id,
                model_id=route.model_id,
                route_id=route.route_id,
            ),
            provider_id=route.provider_id,
            gateway=route.gateway,
            route_id=route.route_id,
            model_id=route.model_id,
            display_name=route.display_name,
            approved=approved,
            available=approved and router.omniroute_settings.ready_for_requests,
            blocked=not approved,
            block_reason=reason,
            capabilities=[item.value for item in route.capabilities],
            task_categories=[item.value for item in route.task_categories],
            context_window=route.context_window,
            locality=route.locality.value,
            privacy_class=route.privacy_class.value,
            streaming=route.supports_streaming,
            tools=route.supports_tools,
            structured_output=route.supports_structured_output,
            expected_latency=route.expected_latency_class,
            quality_tier=route.expected_quality_class,
            development_only=route.development_only,
            terms_status=route.terms_status,
        )

    @staticmethod
    def _route_status(
        router: IntelligenceRouter, route: OmniRouteRoute
    ) -> tuple[str, ProviderTopologyState]:
        if not route.enabled:
            return "route_disabled", ProviderTopologyState.DISABLED
        if route.terms_status != "approved":
            return "terms_not_approved", ProviderTopologyState.BLOCKED
        if route.locality == RouteLocality.CLOUD and not router.omniroute_settings.allow_cloud:
            return "cloud_disabled", ProviderTopologyState.BLOCKED
        if not router.omniroute_settings.api_key:
            return "credential_missing", ProviderTopologyState.CREDENTIAL_MISSING
        if not router.omniroute_settings.enabled:
            return "gateway_disabled", ProviderTopologyState.DISABLED
        if not router.omniroute_registry.is_operator_approved(route):
            return "not_allowlisted", ProviderTopologyState.BLOCKED
        return "allowed", ProviderTopologyState.AVAILABLE

    def _reconcile_catalog(
        self, providers: list[ProviderTopologyNode]
    ) -> ProviderCatalogDiff | None:
        current = {item.topology_node_id: item for item in providers}
        with self._lock:
            previous = self._catalog
            if not previous:
                self._catalog = {key: value.model_copy(deep=True) for key, value in current.items()}
                return None
            added = sorted(set(current) - set(previous))
            removed = sorted(set(previous) - set(current))
            updated = sorted(
                key
                for key in set(current) & set(previous)
                if current[key].model_dump(mode="json")
                != previous[key].model_dump(mode="json")
            )
            previous_models = {
                model.topology_model_id
                for provider in previous.values()
                for model in provider.models
            }
            current_models = {
                model.topology_model_id
                for provider in current.values()
                for model in provider.models
            }
            self._catalog = {key: value.model_copy(deep=True) for key, value in current.items()}
        if not (added or removed or updated or previous_models != current_models):
            return None
        diff = ProviderCatalogDiff(
            added_provider_ids=sorted(
                {current[item].provider_id for item in added if item in current}
            ),
            removed_provider_ids=sorted(
                {previous[item].provider_id for item in removed if item in previous}
            ),
            updated_provider_ids=sorted(
                {current[item].provider_id for item in updated if item in current}
            ),
            added_topology_node_ids=added,
            removed_topology_node_ids=removed,
            updated_topology_node_ids=updated,
            added_models=sorted(current_models - previous_models),
            removed_models=sorted(previous_models - current_models),
        )
        self._publish(
            "llm.catalog.changed",
            "The approved provider catalog changed.",
            diff.model_dump(mode="json"),
        )
        for topology_node_id in added:
            provider = current[topology_node_id]
            self._publish(
                "llm.provider.added",
                "A configured provider was added to the routing catalog.",
                {
                    "topology_node_id": topology_node_id,
                    "provider_id": provider.provider_id,
                    "gateway": provider.gateway,
                },
            )
        for topology_node_id in removed:
            provider = previous[topology_node_id]
            self._publish(
                "llm.provider.removed",
                "A provider was removed from the routing catalog.",
                {
                    "topology_node_id": topology_node_id,
                    "provider_id": provider.provider_id,
                    "gateway": provider.gateway,
                },
            )
        for topology_node_id in updated:
            old = previous.get(topology_node_id)
            new = current.get(topology_node_id)
            if old and new and old.health != new.health:
                self._publish(
                    "llm.provider.health.changed",
                    "A configured provider health state changed.",
                    {
                        "topology_node_id": topology_node_id,
                        "provider_id": new.provider_id,
                        "gateway": new.gateway,
                        "previous_health": old.health.value,
                        "health": new.health.value,
                    },
                )
        return diff

    @staticmethod
    def _active_health(
        provider: ProviderTopologyNode, active: ActiveLLMRoute | None
    ) -> ProviderTopologyState:
        if active is None or provider.topology_node_id != active.topology_node_id:
            return provider.health
        if active.state == RouteLifecycleState.CONNECTING:
            return ProviderTopologyState.CONNECTING
        if active.state == RouteLifecycleState.STREAMING:
            return ProviderTopologyState.STREAMING
        if active.state == RouteLifecycleState.FAILED:
            return ProviderTopologyState.FAILED
        return provider.health

    def _publish_route(
        self,
        event_type: str,
        message: str,
        route: ActiveLLMRoute,
        *,
        severity: EventSeverity = EventSeverity.INFO,
        extra: dict | None = None,
    ) -> None:
        metadata = {
            "request_id": route.request_id,
            "source": route.source,
            "task_category": route.task_category,
            "privacy_class": route.privacy_class,
            "gateway": route.gateway,
            "provider_id": route.provider_id,
            "model_id": route.model_id,
            "route_id": route.route_id,
            "topology_node_id": route.topology_node_id,
            "topology_model_id": route.topology_model_id,
            "topology_route_id": route.topology_route_id,
            "route_state": route.state.value,
            "selection_reason_codes": route.selection_reason_codes,
            "fallback_from": route.fallback_from,
            "fallback_reason": route.fallback_reason,
            "time_to_first_token_ms": route.time_to_first_token_ms,
            "total_latency_ms": route.total_latency_ms,
            "chunk_count": route.chunk_count,
            **(extra or {}),
        }
        self._publish(event_type, message, metadata, severity=severity)

    def _publish(
        self,
        event_type: str,
        message: str,
        metadata: dict,
        *,
        severity: EventSeverity = EventSeverity.INFO,
    ) -> None:
        self.event_logger.log_event(
            event_type,
            message,
            category=EventCategory.LLM,
            title="Intelligence routing",
            metadata=metadata,
            severity=severity,
            persist=False,
        )


llm_routing_observatory = LLMRoutingObservatory()
