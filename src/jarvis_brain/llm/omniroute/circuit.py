"""Bounded in-process circuit breaker for the optional cloud gateway."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from threading import RLock
from time import monotonic


class GatewayCircuitState(str, Enum):
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"
    RATE_LIMITED = "rate_limited"
    CIRCUIT_OPEN = "circuit_open"
    RECOVERING = "recovering"


@dataclass
class GatewayCircuitSnapshot:
    state: GatewayCircuitState
    consecutive_failures: int
    request_allowed: bool


class GatewayCircuitBreaker:
    """Fail fast after bounded gateway failures and recover off the hot path."""

    def __init__(
        self,
        *,
        failure_threshold: int = 2,
        recovery_seconds: float = 30.0,
    ) -> None:
        self.failure_threshold = max(1, failure_threshold)
        self.recovery_seconds = max(1.0, recovery_seconds)
        self._failures = 0
        self._state = GatewayCircuitState.HEALTHY
        self._opened_at: float | None = None
        self._lock = RLock()

    def allow_request(self, *, now: float | None = None) -> bool:
        current = monotonic() if now is None else now
        with self._lock:
            if self._state not in {
                GatewayCircuitState.CIRCUIT_OPEN,
                GatewayCircuitState.RATE_LIMITED,
            }:
                return True
            if self._opened_at is None or current - self._opened_at < self.recovery_seconds:
                return False
            self._state = GatewayCircuitState.RECOVERING
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failures = 0
            self._opened_at = None
            self._state = GatewayCircuitState.HEALTHY

    def record_failure(self, *, rate_limited: bool = False) -> None:
        with self._lock:
            self._failures += 1
            if rate_limited or self._failures >= self.failure_threshold:
                self._state = (
                    GatewayCircuitState.RATE_LIMITED
                    if rate_limited
                    else GatewayCircuitState.CIRCUIT_OPEN
                )
                self._opened_at = monotonic()
            else:
                self._state = GatewayCircuitState.DEGRADED

    def snapshot(self, *, now: float | None = None) -> GatewayCircuitSnapshot:
        allowed = self.allow_request(now=now)
        with self._lock:
            return GatewayCircuitSnapshot(
                state=self._state,
                consecutive_failures=self._failures,
                request_allowed=allowed,
            )


class GatewayCircuitRegistry:
    """Process-local route circuit registry with deterministic route ownership."""

    _circuits: dict[str, GatewayCircuitBreaker] = {}
    _lock = RLock()

    @classmethod
    def for_route(
        cls,
        route_id: str,
        *,
        failure_threshold: int,
        recovery_seconds: float,
    ) -> GatewayCircuitBreaker:
        with cls._lock:
            circuit = cls._circuits.get(route_id)
            if circuit is None:
                circuit = GatewayCircuitBreaker(
                    failure_threshold=failure_threshold,
                    recovery_seconds=recovery_seconds,
                )
                cls._circuits[route_id] = circuit
            return circuit

    @classmethod
    def clear(cls) -> None:
        with cls._lock:
            cls._circuits.clear()
