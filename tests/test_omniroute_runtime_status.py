from __future__ import annotations

import json
from pathlib import Path

from jarvis_brain.llm.omniroute import runtime_status
from jarvis_brain.llm.omniroute.config import OmniRouteSettings


def test_runtime_status_is_honest_when_state_is_missing(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(tmp_path))

    result = runtime_status.safe_runtime_status()

    assert result["managed"] is False
    assert result["process_status"] == "unknown"


def test_runtime_status_exposes_only_safe_state(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("OMNIROUTE_RUNTIME_DIR", str(tmp_path))
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "status": "healthy",
                "pid": 1234,
                "gateway_authentication": True,
                "model_catalog_readiness": True,
                "approved_route_presence": False,
                "secret": "must-not-escape",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(runtime_status, "_process_running", lambda _pid: True)

    result = runtime_status.safe_runtime_status()

    assert result["managed"] is True
    assert result["process_status"] == "running"
    assert result["gateway_health"] == "healthy"
    assert "secret" not in result


def test_routing_configuration_excludes_dynamic_runtime(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        "jarvis_brain.llm.omniroute.config.safe_runtime_status",
        lambda: {"state_age_seconds": 1.0},
    )
    settings = OmniRouteSettings()

    assert "runtime" in settings.safe_status()
    assert "runtime" not in settings.routing_configuration()
