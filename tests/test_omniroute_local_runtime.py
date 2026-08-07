from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from scripts import omniroute_runtime
from scripts.omniroute_runtime import (
    ProcessReadiness,
    ReadinessProbe,
    ReadinessResult,
    RuntimePaths,
)
from scripts.run_omniroute_sidecar import sanitize_log_line


def _paths(tmp_path: Path) -> RuntimePaths:
    return RuntimePaths(
        source=tmp_path / "source",
        node=tmp_path / "node",
        runtime_dir=tmp_path / "runtime",
        port=20128,
    )


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def sleep(self, seconds: float) -> None:
        self.value += seconds


class FakeResponse:
    def __init__(self, status: int, body: bytes) -> None:
        self.status = status
        self._body = body

    def read(self, _limit: int) -> bytes:
        return self._body


class FakeConnection:
    def __init__(
        self,
        response: FakeResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.response = response
        self.error = error
        self.closed = False

    def request(self, *_args, **_kwargs) -> None:
        if self.error is not None:
            raise self.error

    def getresponse(self) -> FakeResponse:
        assert self.response is not None
        return self.response

    def close(self) -> None:
        self.closed = True


def _mock_process_ready(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stage 1 satisfied: the sidecar is alive and its port is stably listening."""
    monkeypatch.setattr(
        omniroute_runtime,
        "_wait_process_ready",
        lambda *_args, **_kwargs: ProcessReadiness(
            True, "process_ready", 0.01, 2, listening=True
        ),
    )


def _mock_ready_wait(monkeypatch: pytest.MonkeyPatch) -> None:
    """Both stages satisfied."""
    _mock_process_ready(monkeypatch)
    monkeypatch.setattr(
        omniroute_runtime,
        "_wait_ready",
        lambda *_args, **_kwargs: ReadinessResult(
            True, "ready", 0.01, 1, http_status=200
        ),
    )


def test_log_sanitizer_removes_secret_shaped_values() -> None:
    line = (
        "Authorization: Bearer abc123 api_key=secret-value "
        "password='visible-value'\n"
    )
    sanitized = sanitize_log_line(line)
    assert "abc123" not in sanitized
    assert "secret-value" not in sanitized
    assert "visible-value" not in sanitized
    assert sanitized.count("<redacted>") == 3


def test_runtime_paths_default_to_managed_sibling_checkout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "OMNIROUTE_INSTALL_ROOT",
        "OMNIROUTE_SOURCE_DIR",
        "OMNIROUTE_NODE_BIN",
        "OMNIROUTE_RUNTIME_DIR",
        "OMNIROUTE_PORT",
    ):
        monkeypatch.delenv(name, raising=False)
    paths = RuntimePaths.from_environment()
    assert paths.source == (
        omniroute_runtime.PROJECT_ROOT.parent / "Jarvis-OmniRoute" / "source"
    ).resolve()
    assert paths.port == 20128


def test_start_is_idempotent_for_owned_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("1234\n", encoding="ascii")
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime,
        "_source_status",
        lambda _paths: {"commit": omniroute_runtime.EXPECTED_COMMIT},
    )
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, pid: pid == 1234
    )
    _mock_ready_wait(monkeypatch)

    result = omniroute_runtime.start(paths)

    assert result["already_running"] is True
    assert result["started"] is False
    assert result["owned_by_invocation"] is False
    assert result["pid"] == 1234


def test_start_rejects_unowned_port(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime, "_source_status", lambda _paths: {"commit": "pin"}
    )
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: False
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: True)

    monkeypatch.setattr(omniroute_runtime, "_port_owner_pids", lambda _port: (999,))

    with pytest.raises(RuntimeError, match="unrelated process"):
        omniroute_runtime.start(paths)


def test_stop_removes_stale_pid_without_signaling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("4321\n", encoding="ascii")
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: False
    )

    result = omniroute_runtime.stop(paths)

    assert result["stale_pid_removed"] is True
    assert not paths.pid_file.exists()


def test_stop_signals_the_group_so_the_node_child_cannot_be_orphaned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Observed physically: SIGKILL on the wrapper alone left port 20128 held.

    The wrapper failed to drain within the graceful window, was killed, and its
    Node child survived holding the listener — which the next start then reported
    as a foreign port conflict.
    """
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("5555\n", encoding="ascii")
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: True
    )
    # A session leader, exactly as `start_new_session=True` guarantees.
    monkeypatch.setattr(omniroute_runtime.os, "getpgid", lambda pid: pid)
    alive = {"value": True}
    monkeypatch.setattr(
        omniroute_runtime, "_process_exists", lambda _pid: alive["value"]
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)
    groups: list[tuple[int, int]] = []
    monkeypatch.setattr(
        omniroute_runtime.os,
        "killpg",
        lambda group, number: groups.append((group, number))
        or alive.update(value=False),
    )
    monkeypatch.setattr(
        omniroute_runtime.os,
        "kill",
        lambda *_a: pytest.fail("the Node child must not be left behind"),
    )

    result = omniroute_runtime.stop(paths, timeout_seconds=1)

    assert result["stopped"] is True
    assert result["forced"] is False
    assert result["listener_released"] is True
    assert groups == [(5555, signal.SIGTERM)]
    assert not paths.pid_file.exists()


def test_stop_escalates_to_the_group_when_graceful_shutdown_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("5555\n", encoding="ascii")
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: True
    )
    monkeypatch.setattr(omniroute_runtime.os, "getpgid", lambda pid: pid)
    signalled: list[tuple[int, int]] = []
    monkeypatch.setattr(
        omniroute_runtime, "_process_exists", lambda _pid: len(signalled) < 2
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)
    monkeypatch.setattr(
        omniroute_runtime.os,
        "killpg",
        lambda group, number: signalled.append((group, number)),
    )

    result = omniroute_runtime.stop(paths, timeout_seconds=1)

    assert result["forced"] is True
    # SIGKILL also goes to the group, so the child dies with the wrapper.
    assert signalled == [(5555, signal.SIGTERM), (5555, signal.SIGKILL)]


def test_stop_never_signals_a_group_the_sidecar_does_not_lead(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If the PID is not its own group leader, only that PID may be signalled."""
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("5555\n", encoding="ascii")
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: True
    )
    monkeypatch.setattr(omniroute_runtime.os, "getpgid", lambda _pid: 1)
    monkeypatch.setattr(
        omniroute_runtime.os,
        "killpg",
        lambda *_a: pytest.fail("an unowned process group must never be signalled"),
    )
    killed: list[int] = []
    monkeypatch.setattr(
        omniroute_runtime, "_process_exists", lambda _pid: not killed
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)
    monkeypatch.setattr(
        omniroute_runtime.os, "kill", lambda pid, _number: killed.append(pid)
    )

    result = omniroute_runtime.stop(paths, timeout_seconds=1)

    assert killed == [5555]
    assert result["stopped"] is True


def test_check_reports_security_and_catalog_without_generation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.runtime_env_file.write_text(
        "OMNIROUTE_API_KEY=test-only-key\n", encoding="utf-8"
    )
    paths.pid_file.write_text("1234\n", encoding="ascii")
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: True
    )
    monkeypatch.setattr(
        omniroute_runtime,
        "_port_open",
        lambda port, timeout=0.15: port == paths.port,
    )
    monkeypatch.setattr(
        omniroute_runtime,
        "_resource_snapshot",
        lambda _pid: {
            "cpu_percent": 0.1,
            "memory_mb": 42.0,
            "process_count": 2,
        },
    )
    monkeypatch.setattr(
        omniroute_runtime,
        "_listener_status",
        lambda _port, _pid: (True, True),
    )

    def fake_http(url: str, *, api_key=None, timeout=2.0):
        del url, timeout
        if api_key:
            return 200, json.dumps({"data": [{"id": "model-1"}]}).encode()
        return 401, b""

    monkeypatch.setattr(omniroute_runtime, "_http_status", fake_http)
    original_project_root = omniroute_runtime.PROJECT_ROOT
    config = tmp_path / "config" / "llm"
    config.mkdir(parents=True)
    (config / "omniroute_routes.yaml").write_text(
        json.dumps({"routes": []}), encoding="utf-8"
    )
    monkeypatch.setattr(omniroute_runtime, "PROJECT_ROOT", tmp_path)

    result = omniroute_runtime.check(paths)

    assert result["status"] == "healthy"
    assert result["gateway_authentication"] is True
    assert result["model_catalog_count"] == 1
    assert result["approved_route_presence"] is False
    assert result["billable_generation_performed"] is False
    assert "test-only-key" not in json.dumps(result)
    monkeypatch.setattr(omniroute_runtime, "PROJECT_ROOT", original_project_root)


def test_state_file_is_owner_only(tmp_path: Path) -> None:
    path = tmp_path / "state.json"
    omniroute_runtime._write_private_json(path, {"status": "ready"})
    assert path.stat().st_mode & 0o077 == 0


def test_root_env_remains_gitignored() -> None:
    result = os.popen("git check-ignore .env").read().strip()
    assert result == ".env"


def test_readiness_becomes_ready_immediately() -> None:
    clock = FakeClock()

    result = omniroute_runtime._wait_ready(
        20128,
        3,
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: ReadinessProbe(True, "ready", 200),
    )

    assert result.ready is True
    assert result.attempts == 1


def test_readiness_becomes_ready_after_several_polls() -> None:
    clock = FakeClock()
    outcomes = iter(
        (
            ReadinessProbe(False, "connection_refused"),
            ReadinessProbe(False, "connection_refused"),
            ReadinessProbe(True, "ready", 200),
        )
    )

    result = omniroute_runtime._wait_ready(
        20128,
        3,
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: next(outcomes),
    )

    assert result.ready is True
    assert result.attempts == 3
    assert result.elapsed_seconds == pytest.approx(0.5)


def test_connection_refused_retries_until_bounded_timeout() -> None:
    clock = FakeClock()

    result = omniroute_runtime._wait_ready(
        20128,
        1,
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: ReadinessProbe(
            False, "connection_refused"
        ),
    )

    assert result.ready is False
    # A port that never accepted is reported as "never listened", not as the
    # ambiguous generic timeout the launcher used to print.
    assert result.reason == "not_listening"
    assert result.listening is False
    assert result.http_responded is False
    assert result.elapsed_seconds == pytest.approx(1.0)
    assert result.attempts == 4


def test_http_accept_without_response_reports_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connection = FakeConnection(error=socket.timeout())
    monkeypatch.setattr(
        omniroute_runtime.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connection,
    )

    result = omniroute_runtime._readiness_probe(
        20128, timeout_seconds=0.1
    )

    assert result.reason == "http_timeout"
    assert connection.closed is True


def _health_body(status: str = "healthy", **extra: object) -> bytes:
    """A payload carrying every field the pinned health contract requires."""
    payload: dict[str, object] = {
        "status": status,
        "timestamp": "2026-07-29T21:58:58.669Z",
        "providerHealth": {},
        "quotaMonitor": {"active": 0},
    }
    payload.update(extra)
    return json.dumps(payload).encode("utf-8")


@pytest.mark.parametrize(
    ("status", "body", "reason"),
    (
        (401, b"{}", "authentication_failure"),
        (403, b"{}", "authentication_failure"),
        (404, b"not found", "not_found"),
        (503, b"{}", "service_unavailable"),
        (500, b"{}", "http_error"),
        (200, b"not-json", "malformed_response"),
        (200, b"[]", "malformed_response"),
        # 200 + JSON object but not the audited health document.
        (200, b'{"status":"healthy"}', "schema_mismatch"),
        # Complete contract, unrecognized status token: a different API.
        (200, _health_body("unhealthy"), "schema_mismatch"),
    ),
)
def test_readiness_probe_classifies_http_failures(
    monkeypatch: pytest.MonkeyPatch,
    status: int,
    body: bytes,
    reason: str,
) -> None:
    connection = FakeConnection(FakeResponse(status, body))
    monkeypatch.setattr(
        omniroute_runtime.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connection,
    )

    result = omniroute_runtime._readiness_probe(
        20128, timeout_seconds=0.1
    )

    assert result.ready is False
    assert result.reason == reason


@pytest.mark.parametrize("report", ("healthy", "degraded"))
def test_compatible_health_contract_is_accepted(
    monkeypatch: pytest.MonkeyPatch, report: str
) -> None:
    """Both documented reports mean the HTTP server is serving.

    `degraded` is the audited route's exception payload — a subsystem signal. It
    must not read as "the process failed to start", which is what previously
    caused the launcher to SIGTERM a live gateway.
    """
    connection = FakeConnection(
        FakeResponse(
            200,
            _health_body(
                report,
                providerSummary={
                    "catalogCount": 290,
                    "configuredCount": 0,
                    "activeCount": 0,
                    "monitoredCount": 0,
                },
            ),
        )
    )
    monkeypatch.setattr(
        omniroute_runtime.http.client,
        "HTTPConnection",
        lambda *_args, **_kwargs: connection,
    )

    result = omniroute_runtime._readiness_probe(20128, timeout_seconds=0.1)

    assert result.ready is True
    assert result.gateway_report == report
    # Route facts travel alongside readiness rather than gating it.
    assert result.providers_active == 0
    assert result.catalog_models == 290


def test_open_port_without_health_response_is_not_a_start_failure() -> None:
    """A gateway still preparing behind an open port is its own state."""
    clock = FakeClock()

    result = omniroute_runtime._wait_ready(
        20128,
        1,
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: ReadinessProbe(False, "http_timeout"),
    )

    assert result.ready is False
    assert result.reason == "health_unresponsive"
    assert result.listening is True
    assert result.http_responded is False


@pytest.mark.parametrize(
    "reason", ("not_found", "schema_mismatch", "malformed_response",
               "authentication_failure"),
)
def test_settled_contract_failures_stop_polling_immediately(reason: str) -> None:
    """Retrying cannot turn a rejected contract into a pass, so do not retry."""
    clock = FakeClock()
    attempts = 0

    def probe(*_args, **_kwargs) -> ReadinessProbe:
        nonlocal attempts
        attempts += 1
        return ReadinessProbe(False, reason, http_status=404)

    result = omniroute_runtime._wait_ready(
        20128, 10, clock=clock, sleep=clock.sleep, probe=probe
    )

    assert result.ready is False
    assert result.reason == reason
    assert attempts == 1


def test_service_unavailable_is_distinct_from_connection_failure() -> None:
    clock = FakeClock()

    unavailable = omniroute_runtime._wait_ready(
        20128, 1, clock=clock, sleep=clock.sleep,
        probe=lambda *_a, **_k: ReadinessProbe(
            False, "service_unavailable", http_status=503
        ),
    )
    clock.value = 0.0
    refused = omniroute_runtime._wait_ready(
        20128, 1, clock=clock, sleep=clock.sleep,
        probe=lambda *_a, **_k: ReadinessProbe(False, "connection_refused"),
    )

    assert unavailable.reason == "service_unavailable"
    assert unavailable.listening is True
    assert unavailable.http_responded is True
    assert refused.reason == "not_listening"
    assert refused.listening is False


def test_child_exit_stops_readiness_polling() -> None:
    class ExitedProcess:
        @staticmethod
        def poll() -> int:
            return 7

    clock = FakeClock()
    result = omniroute_runtime._wait_ready(
        20128,
        30,
        process=ExitedProcess(),
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: pytest.fail("probe must not run"),
    )

    assert result.reason == "process_exit"
    assert result.child_exit_code == 7
    assert result.attempts == 0


def test_cancellation_stops_readiness_polling() -> None:
    class Cancelled:
        @staticmethod
        def is_set() -> bool:
            return True

    clock = FakeClock()
    result = omniroute_runtime._wait_ready(
        20128,
        30,
        cancel_event=Cancelled(),
        clock=clock,
        sleep=clock.sleep,
        probe=lambda *_args, **_kwargs: pytest.fail("probe must not run"),
    )

    assert result.reason == "cancelled"
    assert result.attempts == 0


def test_keyboard_interrupt_cleans_only_new_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime, "_source_status", lambda _paths: {"commit": "pin"}
    )
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: False
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)

    class Child:
        pid = 2468

    monkeypatch.setattr(
        omniroute_runtime.subprocess, "Popen", lambda *_args, **_kwargs: Child()
    )
    _mock_process_ready(monkeypatch)
    monkeypatch.setattr(
        omniroute_runtime,
        "_wait_ready",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(KeyboardInterrupt()),
    )
    terminated: list[int] = []
    monkeypatch.setattr(
        omniroute_runtime,
        "_terminate_owned_process",
        lambda _paths, process: terminated.append(process.pid),
    )

    with pytest.raises(KeyboardInterrupt):
        omniroute_runtime.start(paths, progress=None)

    assert terminated == [2468]
    state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert state["reason"] == "cancelled"


def test_failed_new_child_cleans_pid_and_surfaces_sanitized_tail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.log_file.parent.mkdir(parents=True)
    paths.log_file.write_text(
        "startup failed api_key=super-secret\n", encoding="utf-8"
    )
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime, "_source_status", lambda _paths: {"commit": "pin"}
    )
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, _pid: False
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)

    class Child:
        pid = 1357

    monkeypatch.setattr(
        omniroute_runtime.subprocess, "Popen", lambda *_args, **_kwargs: Child()
    )
    # The child dies during stage 1, before the port is ever bound.
    monkeypatch.setattr(
        omniroute_runtime,
        "_wait_process_ready",
        lambda *_args, **_kwargs: ProcessReadiness(
            False, "process_exit", 0.1, 1, child_exit_code=9
        ),
    )

    def cleanup(_paths, process) -> None:
        assert process.pid == 1357
        paths.pid_file.unlink(missing_ok=True)

    monkeypatch.setattr(omniroute_runtime, "_terminate_owned_process", cleanup)

    with pytest.raises(
        omniroute_runtime.OmniRouteStartupError, match="exit code 9"
    ) as caught:
        omniroute_runtime.start(paths, progress=None)

    assert not paths.pid_file.exists()
    assert "super-secret" not in " ".join(caught.value.log_tail)
    assert "<redacted>" in " ".join(caught.value.log_tail)


def test_timeout_configuration_ranges_are_validated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    monkeypatch.setenv("OMNIROUTE_STARTUP_TIMEOUT_SECONDS", "0")
    with pytest.raises(ValueError, match="between 1 and 300"):
        omniroute_runtime.startup_timeout(paths)
    monkeypatch.setenv("OMNIROUTE_HEALTH_REQUEST_TIMEOUT_SECONDS", "11")
    with pytest.raises(ValueError, match="between 0.1 and 10"):
        omniroute_runtime.health_request_timeout(paths)
    monkeypatch.setenv("OMNIROUTE_HEALTH_RESPONSE_TIMEOUT_SECONDS", "121")
    with pytest.raises(ValueError, match="between 1 and 120"):
        omniroute_runtime.health_response_timeout(paths)


def test_startup_deadline_covers_a_cold_preparation() -> None:
    """The shipped defaults must be able to observe a real cold start.

    Measured on the pinned build: cold TCP bind up to 86 s and a first health
    response beyond 90 s, against the previous 30 s total and 1 s per-request
    budgets. Those budgets could not succeed, so the launcher killed a healthy
    process every time.
    """
    assert omniroute_runtime.DEFAULT_STARTUP_TIMEOUT_SECONDS >= 120.0
    assert omniroute_runtime.DEFAULT_HEALTH_RESPONSE_TIMEOUT_SECONDS >= 30.0
    assert (
        omniroute_runtime.DEFAULT_HEALTH_RESPONSE_TIMEOUT_SECONDS
        > omniroute_runtime.DEFAULT_HEALTH_REQUEST_TIMEOUT_SECONDS
    )


def test_empty_route_pool_is_reported_without_failing_the_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """0 providers / 0 models is a route fact, never a process-start failure."""
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(omniroute_runtime, "_approved_route_ids", lambda _paths: [])
    monkeypatch.setattr(
        omniroute_runtime, "_gateway_model_count", lambda _paths: 0
    )

    readiness = ReadinessResult(
        True, "ready", 1.0, 1, http_status=200, listening=True,
        http_responded=True, gateway_report="healthy",
        providers_configured=0, providers_active=0, catalog_models=290,
    )
    availability = omniroute_runtime.route_availability(paths, readiness)

    assert availability.state == "no_approved_route"
    assert availability.jarvis_route_usable is False
    assert availability.fallback_reason == "no_reviewed_route_approved"
    assert availability.providers_active == 0
    assert availability.gateway_models == 0
    # The catalog is still reported truthfully rather than zeroed out.
    assert availability.catalog_models == 290


def test_route_availability_reports_each_distinct_route_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    monkeypatch.setattr(
        omniroute_runtime, "_approved_route_ids", lambda _paths: ["reviewed-route"]
    )

    def availability(active: int, models: int):
        monkeypatch.setattr(
            omniroute_runtime, "_gateway_model_count", lambda _paths: models
        )
        return omniroute_runtime.route_availability(
            paths,
            ReadinessResult(
                True, "ready", 1.0, 1, http_status=200, listening=True,
                http_responded=True, gateway_report="healthy",
                providers_configured=1, providers_active=active,
                catalog_models=290,
            ),
        )

    assert availability(0, 5).state == "provider_disconnected"
    assert availability(1, 0).state == "catalog_empty"
    usable = availability(1, 5)
    assert usable.state == "available"
    assert usable.jarvis_route_usable is True
    assert usable.fallback_reason is None


def test_checked_in_route_registry_has_no_approved_route() -> None:
    """Guards the reviewed boundary: no route is terms-approved in this repo.

    If this ever fails, a route was enabled and approved without the terms review
    the audit requires.
    """
    assert omniroute_runtime._approved_route_ids(None) == []


def test_start_does_not_spawn_a_second_process_for_a_live_owned_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One managed process per launch; a live owned process is reused, not duplicated."""
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True)
    paths.pid_file.write_text("4242\n", encoding="ascii")
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime, "_source_status", lambda _paths: {"commit": "pin"}
    )
    monkeypatch.setattr(
        omniroute_runtime, "_managed_process", lambda _paths, pid: pid == 4242
    )
    monkeypatch.setattr(
        omniroute_runtime, "_approved_route_ids", lambda _paths: []
    )
    monkeypatch.setattr(
        omniroute_runtime, "_gateway_model_count", lambda _paths: 0
    )
    _mock_ready_wait(monkeypatch)

    spawned: list[object] = []
    monkeypatch.setattr(
        omniroute_runtime.subprocess,
        "Popen",
        lambda *a, **k: spawned.append(a) or (_ for _ in ()).throw(
            AssertionError("a second process was spawned")
        ),
    )
    signalled: list[int] = []
    monkeypatch.setattr(
        omniroute_runtime,
        "_terminate_owned_process",
        lambda _paths, process: signalled.append(process.pid),
    )

    result = omniroute_runtime.start(paths, progress=None)

    assert spawned == []
    # A process this invocation did not create is never terminated.
    assert signalled == []
    assert result["already_running"] is True
    assert result["owned_by_invocation"] is False
    assert result["route_availability"]["state"] == "no_approved_route"


# --- Stage 1: process readiness ----------------------------------------------
#
# Process readiness answers "is the managed sidecar alive and listening", and
# nothing else. It must never be satisfied by a transient bind, and it must never
# be mistaken for route readiness.


def test_process_readiness_requires_consecutive_stable_probes() -> None:
    """A port that opens, closes and reopens was a transient bind, not a start."""
    clock = FakeClock()
    observations = iter((True, False, True, True))

    result = omniroute_runtime._wait_process_ready(
        20128,
        10,
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: next(observations),
    )

    assert result.ready is True
    assert result.reason == "process_ready"
    # The single early bind did not count; stability restarted from zero.
    assert result.attempts == 4


def test_process_readiness_rejects_a_port_that_never_holds() -> None:
    clock = FakeClock()
    observations = iter([True, False] * 200)

    result = omniroute_runtime._wait_process_ready(
        20128,
        1,
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: next(observations),
    )

    assert result.ready is False
    assert result.reason == "unstable_port"
    assert result.listening is True


def test_process_readiness_reports_a_child_that_exits_before_binding() -> None:
    class ExitedProcess:
        @staticmethod
        def poll() -> int:
            return 3

    clock = FakeClock()
    result = omniroute_runtime._wait_process_ready(
        20128,
        30,
        process=ExitedProcess(),
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: pytest.fail("the port must not be probed"),
    )

    assert result.ready is False
    assert result.reason == "process_exit"
    assert result.child_exit_code == 3


def test_process_readiness_rejects_a_listener_that_is_not_the_managed_sidecar() -> None:
    """An open port is not enough: the command identity has to match the pin."""
    clock = FakeClock()

    result = omniroute_runtime._wait_process_ready(
        20128,
        10,
        identity=lambda: False,
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: True,
    )

    assert result.ready is False
    assert result.reason == "process_gone"


def test_cancellation_during_tcp_readiness_stops_immediately() -> None:
    class Cancelled:
        @staticmethod
        def is_set() -> bool:
            return True

    clock = FakeClock()
    result = omniroute_runtime._wait_process_ready(
        20128,
        30,
        cancel_event=Cancelled(),
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: pytest.fail("the port must not be probed"),
    )

    assert result.ready is False
    assert result.reason == "cancelled"
    assert result.attempts == 0


def test_a_long_cold_bind_reports_progress_without_one_line_per_probe() -> None:
    """A 240-second wait must explain itself, but not four times a second."""
    clock = FakeClock()
    lines: list[str] = []

    result = omniroute_runtime._wait_process_ready(
        20128,
        60,
        clock=clock,
        sleep=clock.sleep,
        port_probe=lambda _port: False,
        progress=lines.append,
    )

    assert result.ready is False
    # 60 seconds at four probes a second is 240 iterations; the operator sees a
    # handful of lines, each naming the phase and the elapsed budget.
    assert 5 <= len(lines) <= 8
    assert all("waiting for the loopback port" in line for line in lines)
    assert "60 seconds" in lines[-1]


def test_process_ready_budget_covers_a_measured_cold_bind() -> None:
    # A cold bind was measured at 157.4 s; a budget below it kills the process
    # mid-preparation and guarantees the next launch is cold too.
    assert omniroute_runtime.DEFAULT_PROCESS_READY_TIMEOUT_SECONDS >= 160.0
    # The unified-start route budget is short on purpose: a pending route must not
    # hold the backend and frontend.
    assert omniroute_runtime.DEFAULT_BOOTSTRAP_ROUTE_TIMEOUT_SECONDS <= 15.0
    assert (
        omniroute_runtime.DEFAULT_BOOTSTRAP_ROUTE_TIMEOUT_SECONDS
        < omniroute_runtime.DEFAULT_STARTUP_TIMEOUT_SECONDS
    )


# --- Stage 1 / stage 2 decision table ----------------------------------------


class _FakeChild:
    """A spawned process that stays alive."""

    pid = 5150

    @staticmethod
    def poll() -> None:
        return None


_PROCESS_READY = ProcessReadiness(True, "process_ready", 0.3, 2, listening=True)


def _route(
    reason: str,
    *,
    ready: bool = False,
    report: str | None = None,
    providers_active: int = 0,
):
    return ReadinessResult(
        ready,
        reason,
        1.0,
        2,
        http_status=200 if report else None,
        listening=True,
        http_responded=bool(report),
        gateway_report=report,
        providers_configured=providers_active if report else None,
        providers_active=providers_active if report else None,
        catalog_models=290 if report else None,
    )


def _prepare_start(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    route: ReadinessResult,
    process: ProcessReadiness = _PROCESS_READY,
    managed_pid: int | None = None,
    approved_routes: tuple[str, ...] = (),
    gateway_models: int = 0,
) -> tuple[RuntimePaths, list[int]]:
    paths = _paths(tmp_path)
    paths.runtime_dir.mkdir(parents=True, exist_ok=True)
    if managed_pid is not None:
        paths.pid_file.write_text(f"{managed_pid}\n", encoding="ascii")
    monkeypatch.setattr(omniroute_runtime, "_enabled", lambda _paths: True)
    monkeypatch.setattr(
        omniroute_runtime, "_source_status", lambda _paths: {"commit": "pin"}
    )
    monkeypatch.setattr(
        omniroute_runtime,
        "_managed_process",
        lambda _paths, pid: managed_pid is not None and pid == managed_pid,
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: False)
    monkeypatch.setattr(
        omniroute_runtime, "_approved_route_ids", lambda _paths: list(approved_routes)
    )
    monkeypatch.setattr(
        omniroute_runtime, "_gateway_model_count", lambda _paths: gateway_models
    )
    monkeypatch.setattr(
        omniroute_runtime.subprocess, "Popen", lambda *_a, **_k: _FakeChild()
    )
    monkeypatch.setattr(
        omniroute_runtime, "_wait_process_ready", lambda *_a, **_k: process
    )
    monkeypatch.setattr(omniroute_runtime, "_wait_ready", lambda *_a, **_k: route)
    terminated: list[int] = []
    monkeypatch.setattr(
        omniroute_runtime,
        "_terminate_owned_process",
        lambda _paths, child: terminated.append(child.pid),
    )
    return paths, terminated


def test_immediately_compatible_health_reports_both_stages_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, terminated = _prepare_start(
        tmp_path,
        monkeypatch,
        route=_route("ready", ready=True, report="healthy", providers_active=1),
        approved_routes=("reviewed-route",),
        gateway_models=5,
    )

    result = omniroute_runtime.start(paths, progress=None)

    assert result["status"] == "ready"
    assert result["process_ready"] is True
    assert result["route_ready"] is True
    assert result["health_endpoint"] == "compatible"
    assert result["owned_by_invocation"] is True
    assert result["route_availability"]["state"] == "available"
    assert terminated == []


def test_compatible_degraded_health_is_a_ready_gateway(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`degraded` is the audited exception payload: a subsystem signal, not a failure."""
    paths, terminated = _prepare_start(
        tmp_path,
        monkeypatch,
        route=_route("ready", ready=True, report="degraded", providers_active=1),
        approved_routes=("reviewed-route",),
        gateway_models=5,
    )

    result = omniroute_runtime.start(paths, progress=None)

    assert result["status"] == "ready"
    assert result["route_ready"] is True
    assert result["gateway_report"] == "degraded"
    assert terminated == []


def test_compatible_health_without_a_reviewed_route_is_not_route_ready(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The state this repository is actually in.

    The gateway answers its pinned contract, so the process and the health
    endpoint are fine; no route is terms-approved, so Jarvis must not treat it as
    a usable execution route.
    """
    paths, terminated = _prepare_start(
        tmp_path, monkeypatch, route=_route("ready", ready=True, report="healthy")
    )

    result = omniroute_runtime.start(paths, progress=None)

    assert result["status"] == "ready"
    assert result["health_endpoint"] == "compatible"
    assert result["route_ready"] is False
    assert result["route_availability"]["state"] == "no_approved_route"
    assert terminated == []


@pytest.mark.parametrize(
    "reason", ("health_unresponsive", "http_timeout", "service_unavailable")
)
def test_pending_route_health_preserves_a_listening_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    """The defect this correction exists for.

    A live sidecar that has not answered its health endpoint within the bootstrap
    budget must be reported truthfully and left running -- never terminated, and
    never claimed as a usable route.
    """
    paths, terminated = _prepare_start(tmp_path, monkeypatch, route=_route(reason))

    result = omniroute_runtime.start(paths, progress=None, timeout_seconds=2.0)

    assert result["status"] == "starting"
    assert result["process_ready"] is True
    assert result["route_ready"] is False
    assert result["health_endpoint"] == "pending"
    assert result["reason"] == reason
    assert result["pid"] == _FakeChild.pid
    assert result["owned_by_invocation"] is True
    # The process is alive and owned; it is not killed for being slow.
    assert terminated == []
    availability = result["route_availability"]
    assert availability["state"] == "route_health_pending"
    assert availability["jarvis_route_usable"] is False
    # Unknown counts stay unknown rather than being reported as zero.
    assert availability["providers_active"] is None
    assert availability["gateway_models"] is None
    state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert state["status"] == "starting"
    assert state["route_ready"] is False


@pytest.mark.parametrize(
    "reason",
    ("malformed_response", "schema_mismatch", "not_found", "authentication_failure"),
)
def test_settled_contract_failure_never_becomes_a_usable_route(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, reason: str
) -> None:
    paths, terminated = _prepare_start(tmp_path, monkeypatch, route=_route(reason))

    with pytest.raises(omniroute_runtime.OmniRouteStartupError) as caught:
        omniroute_runtime.start(paths, progress=None)

    assert caught.value.reason == reason
    # Existing security policy for a process this invocation owns.
    assert terminated == [_FakeChild.pid]
    state = json.loads(paths.state_file.read_text(encoding="utf-8"))
    assert state["status"] == "unavailable"


def test_cancellation_during_health_probing_cleans_only_the_new_child(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, terminated = _prepare_start(
        tmp_path, monkeypatch, route=_route("cancelled")
    )

    with pytest.raises(omniroute_runtime.StartupCancelled):
        omniroute_runtime.start(paths, progress=None)

    assert terminated == [_FakeChild.pid]


def test_pre_existing_route_pending_process_is_adopted_truthfully(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, terminated = _prepare_start(
        tmp_path,
        monkeypatch,
        route=_route("health_unresponsive"),
        managed_pid=7777,
    )

    result = omniroute_runtime.start(paths, progress=None, timeout_seconds=2.0)

    assert result["status"] == "starting"
    assert result["process_ready"] is True
    assert result["route_ready"] is False
    assert result["already_running"] is True
    assert result["owned_by_invocation"] is False
    assert result["pid"] == 7777
    # A process this invocation did not create is never terminated.
    assert terminated == []


def test_a_later_start_recognizes_a_now_compatible_pre_existing_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Route readiness is re-evaluated on every start; it is not cached as failed."""
    paths, terminated = _prepare_start(
        tmp_path,
        monkeypatch,
        route=_route("ready", ready=True, report="healthy", providers_active=1),
        managed_pid=7777,
        approved_routes=("reviewed-route",),
        gateway_models=5,
    )
    spawned: list[object] = []
    monkeypatch.setattr(
        omniroute_runtime.subprocess,
        "Popen",
        lambda *a, **_k: spawned.append(a),
    )

    result = omniroute_runtime.start(paths, progress=None)

    assert result["status"] == "ready"
    assert result["route_ready"] is True
    assert result["already_running"] is True
    assert result["owned_by_invocation"] is False
    assert spawned == []
    assert terminated == []


def test_unowned_port_is_reported_as_a_conflict_not_adopted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, terminated = _prepare_start(
        tmp_path, monkeypatch, route=_route("ready", ready=True, report="healthy")
    )
    monkeypatch.setattr(omniroute_runtime, "_port_open", lambda _port: True)
    monkeypatch.setattr(omniroute_runtime, "_port_owner_pids", lambda _port: (4321,))

    with pytest.raises(omniroute_runtime.OmniRouteStartupError) as caught:
        omniroute_runtime.start(paths, progress=None)

    assert caught.value.reason == "port_conflict"
    assert terminated == []


# --- Real process / real HTTP integration -----------------------------------
#
# These exercise the readiness probe against an actual child process serving on
# an actual loopback socket, which is where the shipped bug lived: the port was
# open while the app was still preparing, so nothing responded. Only the gateway
# itself is substituted; the process, socket and HTTP framing are real.

_SERVER_SOURCE = textwrap.dedent(
    """
    import json, sys, time
    from http.server import BaseHTTPRequestHandler, HTTPServer

    MODE = sys.argv[1]
    DELAY = float(sys.argv[2])
    PAYLOAD = {
        "status": "healthy",
        "timestamp": "2026-07-29T00:00:00.000Z",
        "providerHealth": {},
        "quotaMonitor": {"active": 0},
        "providerSummary": {
            "catalogCount": 290, "configuredCount": 0,
            "activeCount": 0, "monitoredCount": 0,
        },
    }

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_a):
            pass

        def do_GET(self):
            # Emulate "port open, app still preparing".
            if DELAY:
                time.sleep(DELAY)
            if MODE == "notfound":
                self.send_response(404)
                self.end_headers()
                self.wfile.write(b"no such route")
                return
            if MODE == "hang":
                time.sleep(30)
                return
            body = b"{}" if MODE == "badschema" else json.dumps(PAYLOAD).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = HTTPServer(("127.0.0.1", int(sys.argv[3])), Handler)
    print("listening", flush=True)
    server.serve_forever()
    """
)


def _free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


class _LocalGateway:
    """A real child process serving a real loopback HTTP socket."""

    def __init__(self, mode: str, *, delay: float = 0.0) -> None:
        self.port = _free_port()
        self.process = subprocess.Popen(
            [sys.executable, "-c", _SERVER_SOURCE, mode, str(delay), str(self.port)],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                with socket.create_connection(("127.0.0.1", self.port), timeout=0.2):
                    return
            except OSError:
                if self.process.poll() is not None:
                    raise AssertionError("local gateway exited during startup")
                time.sleep(0.05)
        raise AssertionError("local gateway never bound")

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
            except (ProcessLookupError, PermissionError):
                pass
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait(timeout=10)


@pytest.fixture()
def local_gateway():
    servers: list[_LocalGateway] = []

    def factory(mode: str, *, delay: float = 0.0) -> _LocalGateway:
        server = _LocalGateway(mode, delay=delay)
        servers.append(server)
        return server

    yield factory
    for server in servers:
        server.close()


def test_integration_real_compatible_gateway_is_accepted(local_gateway) -> None:
    server = local_gateway("healthy")

    result = omniroute_runtime._wait_ready(
        server.port, 30.0, request_timeout_seconds=1.0,
        response_timeout_seconds=10.0, progress=None,
    )

    assert result.ready is True
    assert result.http_status == 200
    assert result.gateway_report == "healthy"
    assert result.providers_active == 0
    assert result.catalog_models == 290


def test_integration_slow_first_response_still_reaches_ready(local_gateway) -> None:
    """The exact shipped failure: a first response slower than the old 1 s budget.

    With the previous single 1 s per-request budget this timed out forever. The
    response budget now applies once the socket is up.
    """
    server = local_gateway("healthy", delay=2.5)

    late = omniroute_runtime._wait_ready(
        server.port, 30.0, request_timeout_seconds=1.0,
        response_timeout_seconds=10.0, progress=None,
    )
    starved = omniroute_runtime._wait_ready(
        server.port, 4.0, request_timeout_seconds=1.0,
        response_timeout_seconds=1.0, progress=None,
    )

    assert late.ready is True
    assert late.gateway_report == "healthy"
    # Same live server, only the response budget differs.
    assert starved.ready is False
    assert starved.reason == "health_unresponsive"
    assert starved.listening is True


def test_integration_real_404_gateway_is_rejected(local_gateway) -> None:
    server = local_gateway("notfound")

    result = omniroute_runtime._wait_ready(
        server.port, 10.0, request_timeout_seconds=1.0,
        response_timeout_seconds=5.0, progress=None,
    )

    assert result.ready is False
    assert result.reason == "not_found"
    assert result.http_status == 404


def test_integration_real_schema_mismatch_is_rejected(local_gateway) -> None:
    server = local_gateway("badschema")

    result = omniroute_runtime._wait_ready(
        server.port, 10.0, request_timeout_seconds=1.0,
        response_timeout_seconds=5.0, progress=None,
    )

    assert result.ready is False
    assert result.reason == "schema_mismatch"
    assert result.http_status == 200


def test_integration_open_port_that_never_answers(local_gateway) -> None:
    server = local_gateway("hang")

    result = omniroute_runtime._wait_ready(
        server.port, 4.0, request_timeout_seconds=1.0,
        response_timeout_seconds=1.5, progress=None,
    )

    assert result.ready is False
    assert result.reason == "health_unresponsive"
    assert result.listening is True
    assert result.http_responded is False
    # Distinguished from a port that was never there at all.
    assert result.reason != "not_listening"


def test_integration_nothing_listening_is_not_listening() -> None:
    port = _free_port()

    result = omniroute_runtime._wait_ready(
        port, 1.5, request_timeout_seconds=0.3,
        response_timeout_seconds=5.0, progress=None,
    )

    assert result.ready is False
    assert result.reason == "not_listening"
    assert result.listening is False


def test_integration_process_is_ready_while_route_health_hangs(local_gateway) -> None:
    """The two stages disagree on a real socket, and that is the whole point.

    The process is up and listening; the health endpoint never answers. Stage 1
    passes, stage 2 stays pending, and neither result contradicts the other.
    """
    server = local_gateway("hang")

    process_state = omniroute_runtime._wait_process_ready(
        server.port, 10.0, progress=None
    )
    route_state = omniroute_runtime._wait_ready(
        server.port, 2.0, request_timeout_seconds=0.5,
        response_timeout_seconds=1.0, progress=None,
    )

    assert process_state.ready is True
    assert process_state.reason == "process_ready"
    assert route_state.ready is False
    assert route_state.reason == "health_unresponsive"


def test_integration_process_readiness_times_out_on_a_dead_port() -> None:
    port = _free_port()

    result = omniroute_runtime._wait_process_ready(port, 1.0, progress=None)

    assert result.ready is False
    assert result.reason == "not_listening"
    assert result.listening is False
    assert result.elapsed_seconds < 5.0
