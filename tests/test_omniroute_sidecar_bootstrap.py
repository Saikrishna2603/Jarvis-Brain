from __future__ import annotations

import stat
from pathlib import Path

import pytest

from scripts.bootstrap_omniroute_sidecar import PROJECT_ROOT, bootstrap


def test_bootstrap_writes_private_files_without_nvidia_secret(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"

    paths = bootstrap(runtime_dir)

    assert all(stat.S_IMODE(path.stat().st_mode) == 0o600 for path in paths)
    assert stat.S_IMODE(runtime_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((runtime_dir / "data").stat().st_mode) == 0o700
    runtime_text = (runtime_dir / "runtime.env").read_text()
    jarvis_text = (runtime_dir / "jarvis.env").read_text()
    assert "NVIDIA" not in runtime_text
    assert "NVIDIA" not in jarvis_text
    assert "OMNIROUTE_API_KEY=or_" in runtime_text
    assert "OMNIROUTE_API_KEY=or_" in jarvis_text
    assert "OMNIROUTE_ALLOW_INTERNAL_FALLBACK=false" in jarvis_text
    assert "OMNIROUTE_ALLOW_COMBOS=false" in jarvis_text


def test_bootstrap_does_not_overwrite_existing_secrets(tmp_path: Path) -> None:
    runtime_dir = tmp_path / "runtime"
    bootstrap(runtime_dir)

    with pytest.raises(FileExistsError):
        bootstrap(runtime_dir)


def test_bootstrap_rejects_repository_destination() -> None:
    with pytest.raises(ValueError, match="outside"):
        bootstrap(PROJECT_ROOT / ".omniroute-test")
