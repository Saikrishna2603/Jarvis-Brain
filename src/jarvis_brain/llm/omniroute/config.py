import ipaddress
import os
from pathlib import Path
from urllib.parse import urlsplit

from jarvis_platform.config import load_app_environment
from jarvis_brain.llm.omniroute.runtime_status import safe_runtime_status
from jarvis_brain.service_paths import SERVICE_ROOT

AUDITED_OMNIROUTE_VERSION = "3.8.49"
AUDITED_OMNIROUTE_COMMIT = "066e9275c45f11ac8b42eb6b807c845528982552"


class OmniRouteConfigurationError(ValueError):
    """Raised when a gateway setting violates Jarvis's network boundary."""


class OmniRouteSettings:
    """Environment-backed OmniRoute settings with fail-closed defaults."""

    def __init__(self) -> None:
        # Constructed on demand, long after every service has imported, so the
        # root has to be named: unqualified this reads the last importer's.
        load_app_environment(SERVICE_ROOT)
        self.enabled = _truthy(os.getenv("OMNIROUTE_ENABLED", "false"))
        self.base_url = _validate_loopback_url(
            os.getenv("OMNIROUTE_BASE_URL", "http://127.0.0.1:20128/v1")
        )
        self.api_key = (os.getenv("OMNIROUTE_API_KEY") or "").strip()
        self.timeout_seconds = _positive_int("OMNIROUTE_TIMEOUT_SECONDS", 60)
        self.connect_timeout_seconds = _positive_int(
            "OMNIROUTE_CONNECT_TIMEOUT_SECONDS", 3
        )
        self.streaming_enabled = _truthy(
            os.getenv("OMNIROUTE_STREAMING_ENABLED", "true")
        )
        self.discovery_enabled = _truthy(
            os.getenv("OMNIROUTE_DISCOVERY_ENABLED", "true")
        )
        self.allow_cloud = _truthy(os.getenv("OMNIROUTE_ALLOW_CLOUD", "false"))
        self.allow_combos = _truthy(os.getenv("OMNIROUTE_ALLOW_COMBOS", "false"))
        self.allow_internal_fallback = _truthy(
            os.getenv("OMNIROUTE_ALLOW_INTERNAL_FALLBACK", "false")
        )
        self.allow_compression = _truthy(
            os.getenv("OMNIROUTE_ALLOW_COMPRESSION", "false")
        )
        self.allowed_providers = _csv("OMNIROUTE_ALLOWED_PROVIDERS")
        self.allowed_models = _csv("OMNIROUTE_ALLOWED_MODELS")
        self.allowed_routes = _csv("OMNIROUTE_ALLOWED_ROUTES")
        self.max_response_bytes = _positive_int(
            "OMNIROUTE_MAX_RESPONSE_BYTES", 4 * 1024 * 1024
        )
        self.healthcheck_seconds = _positive_int(
            "OMNIROUTE_HEALTHCHECK_SECONDS", 30
        )
        self.circuit_failure_threshold = _positive_int(
            "OMNIROUTE_CIRCUIT_FAILURE_THRESHOLD", 2
        )
        self.circuit_recovery_seconds = _positive_int(
            "OMNIROUTE_CIRCUIT_RECOVERY_SECONDS", 30
        )
        configured_path = os.getenv("OMNIROUTE_ROUTE_REGISTRY")
        # Brain's route registry lives in Brain's repository. `PROJECT_ROOT`
        # used to name it and resolved to `jarvis-core` in the composed
        # workspace, because that global is a working-directory guess made
        # before any service declared itself.
        self.route_registry_path = Path(configured_path).expanduser() if configured_path else (
            SERVICE_ROOT / "config" / "llm" / "omniroute_routes.yaml"
        )

    @property
    def endpoint_scope(self) -> str:
        return "loopback"

    @property
    def ready_for_requests(self) -> bool:
        return bool(self.enabled and self.api_key)

    def safe_status(self) -> dict[str, object]:
        return {
            **self.routing_configuration(),
            "runtime": safe_runtime_status(),
        }

    def routing_configuration(self) -> dict[str, object]:
        """Return stable policy inputs without live process-state fields."""
        return {
            "enabled": self.enabled,
            "configured": bool(self.api_key),
            "endpoint_scope": self.endpoint_scope,
            "streaming_enabled": self.streaming_enabled,
            "discovery_enabled": self.discovery_enabled,
            "cloud_allowed": self.allow_cloud,
            "internal_fallback_allowed": self.allow_internal_fallback,
            "combos_allowed": self.allow_combos,
            "compression_allowed": self.allow_compression,
            "audited_source_version": AUDITED_OMNIROUTE_VERSION,
            "audited_source_commit": AUDITED_OMNIROUTE_COMMIT,
        }


def _validate_loopback_url(value: str) -> str:
    parsed = urlsplit((value or "").strip())
    if parsed.scheme != "http":
        raise OmniRouteConfigurationError("OmniRoute must use loopback HTTP.")
    if parsed.username or parsed.password or parsed.query or parsed.fragment:
        raise OmniRouteConfigurationError(
            "OmniRoute URLs cannot contain credentials, query strings, or fragments."
        )
    if not parsed.hostname or parsed.port is None:
        raise OmniRouteConfigurationError("OmniRoute URL requires a loopback host and port.")
    hostname = parsed.hostname.lower()
    is_loopback = hostname == "localhost"
    if not is_loopback:
        try:
            is_loopback = ipaddress.ip_address(hostname).is_loopback
        except ValueError:
            is_loopback = False
    if not is_loopback:
        raise OmniRouteConfigurationError("Remote OmniRoute URLs are disabled.")
    normalized_path = parsed.path.rstrip("/") or "/v1"
    if normalized_path != "/v1":
        raise OmniRouteConfigurationError("OmniRoute base URL must end at /v1.")
    return f"http://{parsed.hostname}:{parsed.port}/v1"


def _truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _positive_int(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default


def _csv(name: str) -> frozenset[str]:
    return frozenset(
        item.strip() for item in (os.getenv(name) or "").split(",") if item.strip()
    )
