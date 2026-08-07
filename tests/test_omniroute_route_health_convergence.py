"""Post-timeout route-health convergence (P0, 2026-08-06).

Physically observed on the pinned 3.8.49 build: the sidecar accepted loopback
connections 1.27 s after spawn, the launcher's 8 s stage-2 budget expired, and
`/api/monitoring/health` first answered 3.4 s *later* -- `healthy`, with one
active provider and a 123-model catalog. `state.json` had already been written
once with `status: starting`, `reason: health_unresponsive`,
`route_health_pending`, and nothing ever re-read it, so every later status
projection reported a gateway that had been healthy for hours as still starting.

These tests pin the convergence contract without a live gateway, a live network
or physical audio: a fake probe supplies the health answer and a fake managed
process supplies liveness.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from scripts import omniroute_runtime
from scripts.omniroute_runtime import ReadinessProbe, RuntimePaths


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        source=tmp_path / "source",
        node=tmp_path / "node",
        runtime_dir=tmp_path / "runtime",
        port=20128,
    )


def _pending_record(paths: RuntimePaths, **overrides: object) -> dict[str, object]:
    """The exact record a unified start writes when stage 2 runs out of budget."""
    record: dict[str, object] = {
        "status": "starting",
        "started": True,
        "already_running": False,
        "owned_by_invocation": True,
        "process_ready": True,
        "route_ready": False,
        "pid": 4242,
        "bind": "127.0.0.1",
        "port": paths.port,
        "loopback_port_listening": True,
        "readiness_url": f"http://127.0.0.1:{paths.port}/api/monitoring/health",
        "readiness_attempts": 2,
        "readiness_elapsed_seconds": 8.002,
        "process_ready_elapsed_seconds": 1.273,
        "route_health_budget_seconds": 8.0,
        "health_endpoint": "pending",
        "reason": "health_unresponsive",
        "gateway_report": None,
        "route_availability": {
            "state": "route_health_pending",
            "providers_configured": None,
            "providers_active": None,
            "catalog_models": None,
            "gateway_models": None,
            "approved_route_present": False,
            "jarvis_route_usable": False,
            "fallback_reason": "route_health_pending",
            "detail": "health_unresponsive",
        },
        "remote_mode": False,
        "tunnels": False,
        "checked_at_epoch": 1_786_073_152.508,
        "version": "3.8.49",
        "commit": "066e9275c45f11ac8b42eb6b807c845528982552",
        "standalone_ready": True,
    }
    record.update(overrides)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    paths.state_file.write_text(json.dumps(record), encoding="utf-8")
    return record


def _healthy_probe(**counts: int) -> ReadinessProbe:
    return ReadinessProbe(
        True,
        "ready",
        http_status=200,
        gateway_report="healthy",
        providers_configured=counts.get("configured", 1),
        providers_active=counts.get("active", 1),
        catalog_models=counts.get("catalog", 290),
    )


def _alive(monkeypatch: pytest.MonkeyPatch, *, alive: bool = True) -> None:
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: alive
    )


# -- the observed failure ---------------------------------------------------


def test_bootstrap_budget_records_a_pending_route_for_a_silent_health_endpoint(
    tmp_path: Path,
) -> None:
    """Stage 2 exhausting its budget behind an open port is `health_unresponsive`.

    Two attempts is arithmetic, not a policy: the first probe gets the 1 s
    request budget because nothing has been observed listening yet, and the
    second gets what is left of the 8 s route budget. This is the state the
    operator saw.
    """
    del tmp_path
    result = omniroute_runtime._wait_ready(
        20128,
        8.0,
        request_timeout_seconds=1.0,
        response_timeout_seconds=60.0,
        clock=(clock := _StepClock()),
        sleep=clock.sleep,
        probe=lambda *_a, **kwargs: clock.consume(kwargs["timeout_seconds"]),
    )

    assert result.ready is False
    assert result.reason == "health_unresponsive"
    assert result.listening is True
    assert result.http_responded is False
    assert result.attempts == 2


class _StepClock:
    """A clock that only advances when a probe or a sleep consumes budget."""

    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds

    def consume(self, timeout_seconds: float) -> ReadinessProbe:
        self.value += timeout_seconds
        return ReadinessProbe(False, "http_timeout")


# -- convergence ------------------------------------------------------------


def test_health_that_arrives_after_the_budget_converges_without_a_restart(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The real defect: a healthy gateway must stop being reported as starting."""
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch)
    monkeypatch.setattr(
        omniroute_runtime, "_approved_route_ids", lambda _paths: ["reviewed-route"]
    )
    monkeypatch.setattr(omniroute_runtime, "_gateway_model_count", lambda _paths: 123)

    state = omniroute_runtime.reconcile(paths, probe=lambda *_a, **_k: _healthy_probe())

    assert state is not None
    assert state["status"] == "ready"
    assert state["health_endpoint"] == "compatible"
    assert state["reason"] is None
    assert state["route_ready"] is True
    assert state["route_availability"]["state"] == "available"
    assert state["route_health_reconciled"] is True
    # The record on disk is the converged one: a later reader sees the truth.
    assert json.loads(paths.state_file.read_text(encoding="utf-8"))["status"] == "ready"


def test_convergence_preserves_the_startup_measurement_it_corrects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Converging must not erase the evidence that convergence was needed."""
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch)
    monkeypatch.setattr(omniroute_runtime, "_approved_route_ids", lambda _paths: [])
    monkeypatch.setattr(omniroute_runtime, "_gateway_model_count", lambda _paths: 123)

    state = omniroute_runtime.reconcile(paths, probe=lambda *_a, **_k: _healthy_probe())

    assert state is not None
    assert state["readiness_elapsed_seconds"] == 8.002
    assert state["route_health_budget_seconds"] == 8.0
    assert state["process_ready_elapsed_seconds"] == 1.273
    # One more observation than the two the bootstrap stage made.
    assert state["readiness_attempts"] == 3
    assert "route_health_reconciled_after_seconds" in state
    # Build attestation is carried, never re-derived: reconcile runs no git.
    assert state["commit"] == "066e9275c45f11ac8b42eb6b807c845528982552"
    assert state["standalone_ready"] is True


def test_a_compatible_gateway_with_no_approved_route_is_honestly_not_route_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence settles gateway health; it never invents route approval.

    This is the checked-in repository's real end state, and it is correct: the
    gateway is healthy, no reviewed route is approved, and Jarvis stays on
    direct Ollama.
    """
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch)
    monkeypatch.setattr(omniroute_runtime, "_approved_route_ids", lambda _paths: [])
    monkeypatch.setattr(omniroute_runtime, "_gateway_model_count", lambda _paths: 123)

    state = omniroute_runtime.reconcile(paths, probe=lambda *_a, **_k: _healthy_probe())

    assert state is not None
    assert state["status"] == "ready"
    assert state["route_ready"] is False
    assert state["route_availability"]["state"] == "no_approved_route"
    assert state["route_availability"]["fallback_reason"] == "no_reviewed_route_approved"


# -- refusals ---------------------------------------------------------------


def test_a_permanently_unhealthy_provider_stays_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A probe that still fails leaves the record byte-for-byte alone.

    Rewriting it with a fresh `checked_at_epoch` would manufacture freshness for
    an observation that has not changed.
    """
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch)
    before = paths.state_file.read_text(encoding="utf-8")

    for _ in range(5):
        state = omniroute_runtime.reconcile(
            paths, probe=lambda *_a, **_k: ReadinessProbe(False, "http_timeout")
        )
        assert state is not None
        assert state["status"] == "starting"
        assert state["route_ready"] is False

    assert paths.state_file.read_text(encoding="utf-8") == before


def test_no_false_ready_is_emitted_for_a_dead_provider(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A record whose process is gone is never probed and never promoted."""
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch, alive=False)
    probes: list[int] = []

    def probe(*_args: object, **_kwargs: object) -> ReadinessProbe:
        probes.append(1)
        return _healthy_probe()

    state = omniroute_runtime.reconcile(paths, probe=probe)

    assert probes == []
    assert state is not None
    assert state["status"] == "starting"
    assert state["route_ready"] is False


@pytest.mark.parametrize("status", ["ready", "stopped", "unavailable", "disabled"])
def test_a_settled_record_is_left_alone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, status: str
) -> None:
    """Only a route-pending record gets a second look.

    Re-probing `ready` would let a single transient blip demote a working
    gateway; re-probing a settled failure would re-litigate a finished answer.
    """
    paths = _paths(tmp_path)
    _pending_record(paths, status=status)
    _alive(monkeypatch)
    probes: list[int] = []

    result = omniroute_runtime.reconcile(
        paths,
        probe=lambda *_a, **_k: (probes.append(1), _healthy_probe())[1],
    )

    assert result is None
    assert probes == []


def test_reconcile_never_spawns_or_signals_a_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Convergence is an observation. Lifecycle stays the launcher's alone."""
    paths = _paths(tmp_path)
    _pending_record(paths)
    _alive(monkeypatch)
    monkeypatch.setattr(omniroute_runtime, "_approved_route_ids", lambda _paths: [])
    monkeypatch.setattr(omniroute_runtime, "_gateway_model_count", lambda _paths: 5)

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("reconcile must not change process lifecycle")

    monkeypatch.setattr(omniroute_runtime.subprocess, "Popen", refuse)
    monkeypatch.setattr(omniroute_runtime, "_signal_sidecar", refuse)
    monkeypatch.setattr(omniroute_runtime, "_terminate_owned_process", refuse)

    assert omniroute_runtime.reconcile(paths, probe=lambda *_a, **_k: _healthy_probe())
    # The PID file is untouched, so the launcher still owns exactly what it started.
    assert not paths.pid_file.exists()


def test_missing_or_malformed_records_are_not_invented(tmp_path: Path) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)

    assert omniroute_runtime.reconcile(paths) is None
    paths.state_file.write_text("not json", encoding="utf-8")
    assert omniroute_runtime.reconcile(paths) is None
    paths.state_file.write_text("[]", encoding="utf-8")
    assert omniroute_runtime.reconcile(paths) is None


# -- exactly one owner ------------------------------------------------------


def test_status_reads_have_exactly_one_reconciliation_owner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Concurrent status reads must not become concurrent probe loops."""
    from jarvis_brain.llm.omniroute import runtime_status

    monkeypatch.setattr(runtime_status, "_reconciled_at_monotonic", None)
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(_paths(tmp_path).runtime_dir))
    started = threading.Event()
    release = threading.Event()
    calls: list[int] = []

    def slow_reconcile(_paths: object) -> None:
        calls.append(1)
        started.set()
        release.wait(timeout=5)

    monkeypatch.setattr(omniroute_runtime, "reconcile", slow_reconcile)

    holder = threading.Thread(target=runtime_status._converge_pending_route_health)
    holder.start()
    assert started.wait(timeout=5)
    # While one reconciliation is in flight, every other caller returns at once
    # rather than queueing behind it or starting a second one.
    runtime_status._converge_pending_route_health()
    runtime_status._converge_pending_route_health()
    release.set()
    holder.join(timeout=5)

    assert calls == [1]


def test_reconciliation_is_rate_limited_so_reads_cannot_become_a_probe_storm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis_brain.llm.omniroute import runtime_status

    monkeypatch.setattr(runtime_status, "_reconciled_at_monotonic", None)
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(_paths(tmp_path).runtime_dir))
    calls: list[int] = []
    monkeypatch.setattr(
        omniroute_runtime, "reconcile", lambda _paths: calls.append(1)
    )

    now = [1_000.0]
    monkeypatch.setattr(runtime_status.time, "monotonic", lambda: now[0])

    for _ in range(10):
        runtime_status._converge_pending_route_health()
    assert calls == [1]

    now[0] += runtime_status.RECONCILE_INTERVAL_SECONDS + 0.1
    runtime_status._converge_pending_route_health()
    assert calls == [1, 1]


def test_a_reconciliation_failure_never_breaks_a_status_read(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from jarvis_brain.llm.omniroute import runtime_status

    monkeypatch.setattr(runtime_status, "_reconciled_at_monotonic", None)
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(_paths(tmp_path).runtime_dir))

    def explode(_paths: object) -> None:
        raise OSError("gateway unreachable")

    monkeypatch.setattr(omniroute_runtime, "reconcile", explode)

    status = runtime_status.safe_runtime_status()

    assert status["managed"] is False
    assert status["process_status"] == "unknown"


def test_status_projection_reports_brain_route_readiness_separately(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Process health and Brain-route readiness are different questions."""
    from jarvis_brain.llm.omniroute import runtime_status

    paths = _paths(tmp_path)
    _pending_record(
        paths,
        status="ready",
        route_ready=False,
        health_endpoint="compatible",
        reason=None,
        route_availability={
            "state": "no_approved_route",
            "providers_configured": 1,
            "providers_active": 1,
            "catalog_models": 290,
            "gateway_models": 123,
            "approved_route_present": False,
            "jarvis_route_usable": False,
            "fallback_reason": "no_reviewed_route_approved",
            "detail": None,
        },
    )
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(paths.runtime_dir))
    monkeypatch.setattr(
        runtime_status, "_converge_pending_route_health", lambda: None
    )

    status = runtime_status.safe_runtime_status()

    assert status["gateway_health"] == "ready"
    assert status["route_ready"] is False
    assert status["route_state"] == "no_approved_route"
    assert status["route_fallback_reason"] == "no_reviewed_route_approved"
