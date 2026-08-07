"""Where Jarvis Brain's own files are.

One constant, derived from this package's location and never reassigned. That
is the whole design, and the reason it exists is worth keeping:

The workspace hosts all five services in one process. `jarvis_platform.config`
keeps the "current service root" in a module-global that every service's
`app.py` writes at import, so the last import to run owns it — Skills, as it
happens. Anything resolving a path through that global therefore resolves it
against whichever service imported last, not against the service asking.

A shared library cannot infer its caller by looking at itself. A service can,
because it *is* the thing it is looking for. So Brain answers that question
here, and passes the answer explicitly to shared helpers that need it.

Import this rather than `jarvis_platform.config.PROJECT_ROOT`.
"""

from __future__ import annotations

from pathlib import Path

#: `<workspace>/jarvis-brain` — this file is at `<root>/src/jarvis_brain/`.
SERVICE_ROOT: Path = Path(__file__).resolve().parents[2]

__all__ = ["SERVICE_ROOT"]
