"""The Jarvis Brain service application.

Brain's own HTTP surface, extracted from nine of the monolith's route modules.

The health contract here is the one that mattered most to get right. In the
monolith, an unreachable LLM route propagated into a single global verdict and
presented as "Jarvis offline". Here it makes **Brain** `degraded` and nothing
else — Core stays online, Voice keeps listening, and an operator is pointed at
the actual problem.
"""

from __future__ import annotations

import os
import pathlib
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from jarvis_platform.config import load_app_environment, set_service_root

# Declare this repository as the service root before anything reads config.
# The shared platform package cannot infer it — see jarvis_platform.config.
SERVICE_ROOT = set_service_root(pathlib.Path(__file__).resolve().parents[2])
load_app_environment()

from jarvis_brain.ports import unbound_ports
from jarvis_brain.routes.agent_lifecycle import router as agent_lifecycle_router
from jarvis_brain.routes.agent_narration import router as agent_narration_router
from jarvis_brain.routes.agent_stream import router as agent_stream_router
from jarvis_brain.routes.brain import router as brain_router
from jarvis_brain.routes.briefing import router as briefing_router
from jarvis_brain.routes.llm import router as llm_router
from jarvis_brain.routes.planner import router as planner_router
from jarvis_brain.routes.swarm import router as swarm_router
from jarvis_brain.routes.world import router as world_router

SERVICE_VERSION = "0.1.0"
CONTRACT_VERSION = "1.0.0"

FRONTEND_ORIGINS = (
    f"http://localhost:{os.environ.get('JARVIS_FRONTEND_PORT', '5173')}",
    f"http://127.0.0.1:{os.environ.get('JARVIS_FRONTEND_PORT', '5173')}",
    f"http://localhost:{os.environ.get('JARVIS_CORE_PORT', '8000')}",
    f"http://127.0.0.1:{os.environ.get('JARVIS_CORE_PORT', '8000')}",
)


def _provider_report() -> dict[str, object]:
    """Ask the LLM provider whether it can answer. Never raises.

    A provider that is configured but unreachable makes Brain `degraded`, not
    `offline`: the service is running and will recover when the route does.

    **A mock provider is never reported as ready.** During extraction this
    service lost its `.env`, fell back to `MockLLMProvider`, and cheerfully
    answered "LLM provider is available" — a mock presenting as a real model.
    `mocked` is now part of the contract so that cannot happen silently again.
    """

    from jarvis_platform.config import environment_is_loaded

    configured = environment_is_loaded()

    try:
        from jarvis_brain.llm.llm_provider_factory import create_llm_provider

        provider = create_llm_provider(None)
    except Exception as exc:  # noqa: BLE001 - readiness must not crash
        return {
            "ready": False,
            "detail": f"LLM provider could not be created: {exc}",
            "provider": None,
            "mocked": False,
            "environment_loaded": configured,
        }

    if provider is None:
        return {
            "ready": False,
            "detail": "No LLM provider is configured.",
            "provider": None,
            "mocked": False,
            "environment_loaded": configured,
        }

    name = type(provider).__name__
    mocked = "mock" in name.lower()

    try:
        reachable = bool(provider.is_available())
    except Exception as exc:  # noqa: BLE001
        reachable = False
        detail = f"LLM provider health check failed: {exc}"
    else:
        if mocked:
            detail = (
                f"{name} is in use — Brain would answer with fabricated text. "
                "Reporting not-ready rather than pretending to reason."
                + ("" if configured else " No .env was found for this service.")
            )
        elif reachable:
            detail = f"{name} is available."
        else:
            detail = f"{name} is configured but unreachable."

    return {
        "ready": reachable and not mocked,
        "detail": detail,
        "provider": name,
        "mocked": mocked,
        "environment_loaded": configured,
    }


def create_app() -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield

    app = FastAPI(title="Jarvis Brain", version=SERVICE_VERSION, lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(FRONTEND_ORIGINS),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    for router in (
        brain_router,
        agent_lifecycle_router,
        agent_narration_router,
        agent_stream_router,
        briefing_router,
        llm_router,
        planner_router,
        swarm_router,
        world_router,
    ):
        app.include_router(router)

    # -- health (spec §9) ---------------------------------------------------

    @app.get("/health/live")
    async def health_live() -> dict[str, object]:
        return {"live": True}

    @app.get("/health/ready")
    async def health_ready() -> dict[str, object]:
        report = _provider_report()
        return {"ready": report["ready"], "detail": report["detail"]}

    @app.get("/health/status")
    async def health_status() -> dict[str, object]:
        """Full detail, including which sibling collaborators are unwired.

        `unbound_ports` is reported rather than hidden. A Brain with 19 unbound
        ports can still answer a plain question but cannot reach memory or
        skills, and an operator should be able to see that difference without
        reading a stack trace.
        """

        report = _provider_report()
        unbound = unbound_ports()
        return {
            "service": "brain",
            "service_version": SERVICE_VERSION,
            "contract_version": CONTRACT_VERSION,
            **report,
            "unbound_ports": list(unbound),
            "fully_wired": not unbound,
        }

    return app


app = create_app()


def main() -> None:  # pragma: no cover - process entry point
    import uvicorn

    uvicorn.run(
        app,
        host=os.environ.get("JARVIS_BRAIN_HOST", "127.0.0.1"),
        port=int(os.environ.get("JARVIS_BRAIN_PORT", "8102")),
    )


if __name__ == "__main__":  # pragma: no cover
    main()
