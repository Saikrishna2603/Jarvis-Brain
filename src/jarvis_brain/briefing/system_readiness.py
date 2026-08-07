import os
from typing import Any, Callable

from sqlalchemy import text

from jarvis_platform.config import load_app_environment
from jarvis_brain.service_paths import SERVICE_ROOT
from jarvis_platform.integrations.integration_registry import IntegrationRegistry
from jarvis_brain.ports import RetrievalRegistry
from jarvis_brain.ports import LocalMacTTSProvider
from jarvis_brain.ports import create_tts_provider


OK = "ok"
DEGRADED = "degraded"
ERROR = "error"
NOT_CONFIGURED = "not_configured"

DATABASE_CONNECT_TIMEOUT_SECONDS = 2


def _state(name: str, label: str, state: str, detail: str) -> dict[str, Any]:
    """Return one subsystem readiness entry."""
    return {"name": name, "label": label, "state": state, "detail": detail}


_health_engine = None


def _get_health_engine():
    """Return an engine that fails fast instead of hanging the briefing.

    The app's main engine has no connect timeout, so probing a database that is
    simply not running blocks long enough to time the whole section out. A
    greeting must never wait on a dead socket, so health checks get their own
    short-timeout engine.
    """
    global _health_engine
    if _health_engine is None:
        from sqlalchemy import create_engine

        from jarvis_platform.db.session import DATABASE_URL

        _health_engine = create_engine(
            DATABASE_URL,
            connect_args={"connect_timeout": DATABASE_CONNECT_TIMEOUT_SECONDS},
            pool_pre_ping=True,
        )
    return _health_engine


def check_database() -> dict[str, Any]:
    """Check the database with a real query rather than assuming it is up."""
    try:
        with _get_health_engine().connect() as connection:
            connection.execute(text("SELECT 1"))
        return _state("database", "Database", OK, "PostgreSQL is reachable.")
    except Exception as error:  # noqa: BLE001 - a dead database is a report, not a crash
        return _state(
            "database",
            "Database",
            DEGRADED,
            "PostgreSQL is not reachable. Persistent memory is unavailable. "
            f"({type(error).__name__})",
        )


def check_llm() -> dict[str, Any]:
    """Report the configured LLM providers.

    This reports configuration, not liveness. Jarvis does not open a network
    connection to a model provider just to greet you, so the detail says so
    rather than implying the model was reached.
    """
    load_app_environment(SERVICE_ROOT)
    enabled = str(os.getenv("LLM_ENABLED", "")).strip().lower() in {"1", "true", "yes", "on"}
    provider = os.getenv("LLM_PROVIDER", "").strip()
    model = os.getenv("LLM_MODEL", "").strip()

    if not enabled or not provider:
        return _state(
            "llm",
            "LLM providers",
            NOT_CONFIGURED,
            "No LLM provider is enabled. Jarvis is running on rules only.",
        )
    return _state(
        "llm",
        "LLM providers",
        OK,
        f"Provider '{provider}' is configured with model '{model or 'unspecified'}'. "
        "Liveness is not probed at briefing time.",
    )


def check_voice() -> dict[str, Any]:
    """Check whether Jarvis can genuinely speak on this host."""
    provider = create_tts_provider()
    if not isinstance(provider, LocalMacTTSProvider):
        return _state(
            "voice",
            "Voice output",
            NOT_CONFIGURED,
            "Voice output is using the mock provider. Nothing will be spoken aloud.",
        )
    if not provider.is_available():
        return _state(
            "voice",
            "Voice output",
            DEGRADED,
            "Local speech is selected but unavailable on this host.",
        )
    return _state("voice", "Voice output", OK, "Local speech synthesis is available.")


def check_microphone() -> dict[str, Any]:
    """Report microphone and wake-word state.

    There is no microphone capture and no wake-word detector in this system.
    Saying so plainly is the point: Jarvis must never imply it is listening.
    """
    return _state(
        "microphone",
        "Microphone and wake word",
        NOT_CONFIGURED,
        "No microphone capture and no wake-word detection exist. Jarvis is not listening.",
    )


def check_vision() -> dict[str, Any]:
    """Report vision provider state."""
    return _state(
        "vision",
        "Vision",
        NOT_CONFIGURED,
        "Only a mock vision provider exists. Jarvis cannot see.",
    )


def check_integrations(integration_registry: IntegrationRegistry) -> dict[str, Any]:
    """Report whether any integration reads real user data."""
    real = integration_registry.real_connectors()
    if not real:
        return _state(
            "integrations",
            "Integrations",
            NOT_CONFIGURED,
            "No real integration is connected. Mail, calendar, and files are mock connectors only.",
        )
    names = ", ".join(connector.name for connector in real)
    return _state("integrations", "Integrations", OK, f"Connected: {names}.")


def check_retrieval(retrieval_registry: RetrievalRegistry) -> dict[str, Any]:
    """Report retrieval driver state."""
    drivers = retrieval_registry.list_drivers()
    networked = [
        driver for driver in drivers if getattr(driver, "enable_network", False)
    ]
    if not drivers:
        return _state("retrieval", "Retrieval", NOT_CONFIGURED, "No retrieval driver is registered.")
    if not networked:
        return _state(
            "retrieval",
            "Retrieval",
            NOT_CONFIGURED,
            f"{len(drivers)} retrieval drivers are registered, but network access is disabled.",
        )
    return _state(
        "retrieval",
        "Retrieval",
        OK,
        f"{len(networked)} of {len(drivers)} retrieval drivers have network access enabled.",
    )


def check_agent_stream() -> dict[str, Any]:
    """Report whether the live agent event stream is enabled."""
    load_app_environment(SERVICE_ROOT)
    enabled = str(os.getenv("AGENT_STREAM_ENABLED", "true")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if not enabled:
        return _state(
            "sse",
            "Agent event stream",
            NOT_CONFIGURED,
            "The live agent event stream is disabled.",
        )
    return _state("sse", "Agent event stream", OK, "Server-sent agent events are enabled.")


def check_scheduler(scheduled_briefing_enabled: bool = False) -> dict[str, Any]:
    """Report scheduler state.

    Scheduled briefings do not exist yet. This says so rather than implying a
    morning briefing will arrive on its own.
    """
    if not scheduled_briefing_enabled:
        return _state(
            "scheduler",
            "Scheduler",
            NOT_CONFIGURED,
            "No scheduled briefing is configured. Briefings run only when you ask.",
        )
    return _state("scheduler", "Scheduler", OK, "Scheduled briefings are configured.")


def build_system_readiness_provider(
    integration_registry: IntegrationRegistry,
    retrieval_registry: RetrievalRegistry,
) -> Callable[[], list[dict[str, Any]]]:
    """Build the callable the system readiness collector reads from."""

    def provider() -> list[dict[str, Any]]:
        """Return the real state of every subsystem the briefing depends on."""
        return [
            _state("backend", "Backend", OK, "The Jarvis API is serving requests."),
            _state("brain", "Brain engine", OK, "Rule-based decision authority is active."),
            _state("memory", "Memory", OK, "Task, audit, and semantic memory are active."),
            _state("agents", "Agent lifecycle", OK, "Agent lifecycle tracking is active."),
            check_database(),
            check_llm(),
            check_voice(),
            check_microphone(),
            check_vision(),
            check_integrations(integration_registry),
            check_retrieval(retrieval_registry),
            check_agent_stream(),
            check_scheduler(),
        ]

    return provider
