from __future__ import annotations

from jarvis_brain.llm.omniroute.circuit import (
    GatewayCircuitBreaker,
    GatewayCircuitState,
)


def test_circuit_opens_after_bounded_failures_and_fails_fast() -> None:
    circuit = GatewayCircuitBreaker(failure_threshold=2, recovery_seconds=30)

    circuit.record_failure()
    assert circuit.snapshot(now=0).state == GatewayCircuitState.DEGRADED
    circuit.record_failure()

    snapshot = circuit.snapshot(now=0)
    assert snapshot.state == GatewayCircuitState.CIRCUIT_OPEN
    assert snapshot.request_allowed is False


def test_circuit_recovers_after_interval() -> None:
    circuit = GatewayCircuitBreaker(failure_threshold=1, recovery_seconds=10)
    circuit.record_failure()
    opened = circuit._opened_at
    assert opened is not None

    assert circuit.allow_request(now=opened + 5) is False
    assert circuit.allow_request(now=opened + 11) is True
    assert circuit.snapshot(now=opened + 11).state == GatewayCircuitState.RECOVERING

    circuit.record_success()
    assert circuit.snapshot().state == GatewayCircuitState.HEALTHY


def test_rate_limit_opens_without_retry_loop() -> None:
    circuit = GatewayCircuitBreaker(failure_threshold=3, recovery_seconds=30)
    circuit.record_failure(rate_limited=True)

    snapshot = circuit.snapshot()
    assert snapshot.state == GatewayCircuitState.RATE_LIMITED
    assert snapshot.request_allowed is False
