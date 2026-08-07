from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import (
    ModelCapability,
    OmniRouteRoute,
    RouteClassification,
    RouteLocality,
    RoutePrivacy,
    RouteSelection,
    TaskCategory,
)


class OmniRouteSelectionPolicy:
    """Hard-filter and score reviewed routes; privacy and terms are never weights."""

    def __init__(
        self,
        registry: OmniRouteRouteRegistry | None = None,
        settings: OmniRouteSettings | None = None,
        available_model_ids: set[str] | None = None,
        route_health: dict[str, float] | None = None,
        historical_success: dict[str, float] | None = None,
    ) -> None:
        self.settings = settings or (registry.settings if registry else OmniRouteSettings())
        self.registry = registry or OmniRouteRouteRegistry(self.settings)
        self.available_model_ids = available_model_ids
        self.route_health = route_health or {}
        self.historical_success = historical_success or {}

    def select(self, classification: RouteClassification) -> RouteSelection:
        if not self.settings.enabled:
            return RouteSelection(selected=False, reason_codes=["gateway_disabled"])
        if not self.settings.api_key:
            return RouteSelection(selected=False, reason_codes=["gateway_key_missing"])

        rejected: dict[str, list[str]] = {}
        candidates: list[tuple[float, OmniRouteRoute, list[str]]] = []
        for route in self.registry.list_routes():
            failures = self._hard_filter(route, classification)
            if failures:
                rejected[route.route_id] = failures
                continue
            score, reasons = self._score(route, classification)
            candidates.append((score, route, reasons))
        if not candidates:
            return RouteSelection(
                selected=False,
                reason_codes=["no_eligible_gateway_route"],
                rejected_routes=rejected,
            )
        score, route, reasons = sorted(
            candidates, key=lambda item: (-item[0], item[1].route_id)
        )[0]
        return RouteSelection(
            selected=True,
            route=route,
            score=round(score, 4),
            reason_codes=reasons,
            rejected_routes=rejected,
        )

    def _hard_filter(
        self, route: OmniRouteRoute, classification: RouteClassification
    ) -> list[str]:
        failures: list[str] = []
        if not self.registry.is_operator_approved(route):
            failures.append("not_operator_approved")
        if route.locality == RouteLocality.UNKNOWN:
            failures.append("unknown_locality")
        if route.locality == RouteLocality.MIXED:
            failures.append("mixed_route_disabled")
        if route.locality == RouteLocality.CLOUD:
            if not self.settings.allow_cloud:
                failures.append("cloud_disabled")
            if classification.privacy != "cloud_allowed":
                failures.append("request_not_cloud_allowed")
            if route.privacy_class not in {RoutePrivacy.CLOUD_ALLOWED, RoutePrivacy.PUBLIC_ONLY}:
                failures.append("route_privacy_mismatch")
        if classification.privacy == "local_only" and route.locality != RouteLocality.LOCAL:
            failures.append("local_only_violation")
        if route.locality == RouteLocality.LOCAL and route.privacy_class != RoutePrivacy.LOCAL_ONLY:
            failures.append("local_route_privacy_mismatch")
        required = set(classification.required_capabilities)
        provided = set(route.capabilities)
        if not required.issubset(provided):
            failures.append("required_capability_missing")
        if (
            ModelCapability.STRUCTURED_OUTPUT in required
            and not route.supports_structured_output
        ):
            failures.append("structured_output_not_supported")
        if ModelCapability.TOOL_CALLING in required and not route.supports_tools:
            failures.append("tool_calling_not_supported")
        if classification.estimated_context_tokens > route.context_window:
            failures.append("context_window_too_small")
        if self.available_model_ids is not None and route.model_id not in self.available_model_ids:
            failures.append("model_not_discovered")
        if self.route_health.get(route.route_id, 1.0) <= 0:
            failures.append("provider_unavailable")
        if classification.task_category in {TaskCategory.SYSTEM_COMMAND, TaskCategory.SAFETY_SENSITIVE}:
            failures.append("deterministic_authority_required")
        return list(dict.fromkeys(failures))

    def _score(
        self, route: OmniRouteRoute, classification: RouteClassification
    ) -> tuple[float, list[str]]:
        required = set(classification.required_capabilities)
        provided = set(route.capabilities)
        capability_match = len(required & provided) / max(1, len(required))
        task_match = 1.0 if classification.task_category in route.task_categories else 0.4
        quality = {"high": 1.0, "balanced": 0.7, "standard": 0.6}.get(
            route.expected_quality_class, 0.4
        )
        latency = {"very_low": 1.0, "low": 0.9, "balanced": 0.65, "high": 0.3}.get(
            route.expected_latency_class, 0.4
        )
        context_fit = min(1.0, route.context_window / max(1, classification.estimated_context_tokens))
        cost = {"free_development": 0.8, "low": 0.8, "medium": 0.6, "high": 0.3}.get(route.cost_class, 0.4)
        quota = {"healthy": 1.0, "limited": 0.5, "unknown": 0.4}.get(route.quota_class, 0.4)
        operator = route.operator_priority / 100
        observed_health = max(0.0, min(1.0, self.route_health.get(route.route_id, 1.0)))
        success_rate = max(
            0.0, min(1.0, self.historical_success.get(route.route_id, 0.5))
        )
        health = (observed_health + success_rate) / 2
        total = (
            capability_match * 0.25
            + task_match * 0.15
            + quality * 0.15
            + health * 0.12
            + latency * 0.10
            + context_fit * 0.08
            + quota * 0.05
            + cost * 0.05
            + operator * 0.05
        )
        return total, [
            "hard_filters_passed",
            f"capability_match:{capability_match:.2f}",
            f"task_match:{task_match:.2f}",
            f"health:{observed_health:.2f}",
            f"historical_success:{success_rate:.2f}",
            f"route:{route.route_id}",
        ]
