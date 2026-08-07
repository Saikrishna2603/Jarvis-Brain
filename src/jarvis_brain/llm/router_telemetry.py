from time import perf_counter

from jarvis_platform.schemas.common import utc_now
from jarvis_platform.schemas.llm import LLMProviderName
from jarvis_platform.schemas.llm_router import LLMRouterStatistics, LLMRoutingDecision


class LLMRouterTelemetry:
    """In-memory router telemetry safe for status APIs."""

    def __init__(self, max_decisions: int = 100) -> None:
        self.max_decisions = max_decisions
        self.stats = LLMRouterStatistics()
        self._latency_total_ms = 0.0

    def start_timer(self) -> float:
        return perf_counter()

    def record_success(self, provider: LLMProviderName, decision: LLMRoutingDecision, started_at: float, fallback_used: bool) -> None:
        self.stats.total_requests += 1
        self.stats.provider_successes[provider.value] = self.stats.provider_successes.get(provider.value, 0) + 1
        if fallback_used:
            self.stats.fallback_count += 1
        self._record_latency(started_at)
        self._record_decision(decision)

    def record_failure(self, provider: LLMProviderName, decision: LLMRoutingDecision | None = None) -> None:
        self.stats.provider_failures[provider.value] = self.stats.provider_failures.get(provider.value, 0) + 1
        if decision:
            self._record_decision(decision)

    def snapshot(self) -> LLMRouterStatistics:
        self.stats.updated_at = utc_now()
        return self.stats

    def _record_latency(self, started_at: float) -> None:
        latency_ms = (perf_counter() - started_at) * 1000
        self._latency_total_ms += latency_ms
        self.stats.average_latency_ms = round(self._latency_total_ms / max(1, self.stats.total_requests), 3)

    def _record_decision(self, decision: LLMRoutingDecision) -> None:
        self.stats.routing_decisions.append(decision.model_dump(mode="json"))
        self.stats.routing_decisions = self.stats.routing_decisions[-self.max_decisions:]


llm_router_telemetry = LLMRouterTelemetry()
