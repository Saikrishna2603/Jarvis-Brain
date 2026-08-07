"""Safe projection of the owner-only local OmniRoute runtime state."""

from __future__ import annotations

import json
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any

# How often a route-pending record may be re-observed. The launcher's stage-2
# budget is short by design, so a gateway that answers a few seconds later is
# recorded as `starting`; without this the record stayed `health_unresponsive`
# for the life of the process. Five seconds is fast enough that a status read
# after a normal start already tells the truth, and slow enough that a busy
# dashboard cannot turn status reads into a probe storm.
RECONCILE_INTERVAL_SECONDS = 5.0

# One reconciliation at a time, process-wide. The lock is taken non-blocking, so
# concurrent status reads never queue behind a probe -- they simply project the
# record as it stands and let the holder finish. This is what keeps there being
# exactly one health-reconciliation owner rather than one per request.
_reconcile_lock = threading.Lock()
_reconciled_at_monotonic: float | None = None


def _converge_pending_route_health() -> None:
    """Give a route-pending record at most one bounded second look.

    Delegates to the launcher, which owns the pinned health contract. Doing the
    probe here would put a second interpretation of that contract in the tree.
    """
    global _reconciled_at_monotonic
    if not _reconcile_lock.acquire(blocking=False):
        return
    try:
        now = time.monotonic()
        if (
            _reconciled_at_monotonic is not None
            and now - _reconciled_at_monotonic < RECONCILE_INTERVAL_SECONDS
        ):
            return
        # Recorded before the probe, so a failing or slow gateway is rate-limited
        # exactly like a succeeding one and cannot be re-probed on every read.
        _reconciled_at_monotonic = now
        from scripts.omniroute_runtime import RuntimePaths, reconcile

        reconcile(RuntimePaths.from_environment())
    except Exception:
        # A status projection must never fail because convergence could not run.
        return
    finally:
        _reconcile_lock.release()


def safe_runtime_status() -> dict[str, Any]:
    """Return process/runtime signals without paths, credentials, or payloads."""
    _converge_pending_route_health()
    runtime_dir = Path(
        os.getenv("OMNIROUTE_RUNTIME_DIR", "~/.jarvis/omniroute")
    ).expanduser()
    state_path = runtime_dir / "state.json"
    try:
        if state_path.stat().st_size > 64 * 1024:
            raise ValueError("Runtime state is too large.")
        payload = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, ValueError, json.JSONDecodeError):
        return {
            "managed": False,
            "process_status": "unknown",
            "gateway_health": "unknown",
            "state_age_seconds": None,
            "remote_mode_disabled": True,
            "tunnel_absent": True,
        }
    if not isinstance(payload, dict):
        return {
            "managed": False,
            "process_status": "unknown",
            "gateway_health": "unknown",
            "state_age_seconds": None,
            "remote_mode_disabled": True,
            "tunnel_absent": True,
        }
    pid = payload.get("pid")
    process_running = _process_running(pid)
    checked_at = payload.get("checked_at_epoch")
    age = (
        max(0.0, time.time() - float(checked_at))
        if isinstance(checked_at, (int, float))
        else None
    )
    return {
        "managed": True,
        "process_status": "running" if process_running else "stopped",
        "gateway_health": str(payload.get("status") or "unknown"),
        "state_age_seconds": round(age, 3) if age is not None else None,
        "remote_mode_disabled": payload.get("remote_mode", False) is False,
        "tunnel_absent": payload.get("tunnel_absent", payload.get("tunnels") is False),
        "loopback_port_listening": bool(
            payload.get("loopback_port_listening", process_running)
        ),
        "gateway_authentication": bool(
            payload.get("gateway_authentication", False)
        ),
        "model_catalog_readiness": bool(
            payload.get("model_catalog_readiness", False)
        ),
        "approved_route_presence": bool(
            payload.get("approved_route_presence", False)
        ),
        # Brain-route readiness, reported separately from process and gateway
        # health because they answer different questions and were previously read
        # as one. A listening sidecar with a compatible health contract and no
        # terms-approved route is honestly not route-ready, and Jarvis stays on
        # direct Ollama -- that is a working system, not an offline one.
        "route_ready": bool(payload.get("route_ready", False)),
        "route_state": _optional_text(_route_field(payload, "state")) or "unknown",
        "route_fallback_reason": _optional_text(
            _route_field(payload, "fallback_reason")
        ),
        "route_health_reconciled": bool(payload.get("route_health_reconciled", False)),
    }


def _route_field(payload: dict[str, Any], name: str) -> object:
    availability = payload.get("route_availability")
    return availability.get(name) if isinstance(availability, dict) else None


def _optional_text(value: object) -> str | None:
    """A short, safe string or nothing. Never a payload, path or credential."""
    return value[:64] if isinstance(value, str) and value else None


def _process_running(pid: object) -> bool:
    if not isinstance(pid, int) or pid <= 1:
        return False
    try:
        os.kill(pid, 0)
    except (ProcessLookupError, PermissionError, OSError):
        return False
    try:
        command = subprocess.run(
            ["/bin/ps", "-p", str(pid), "-o", "command="],
            check=False,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except OSError:
        return False
    return "run_omniroute_sidecar.py" in command
