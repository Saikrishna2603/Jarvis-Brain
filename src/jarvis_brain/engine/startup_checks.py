import os
from typing import Any

from jarvis_platform.config import env_path_for
from jarvis_platform.security.auth_guard import AuthGuard
from jarvis_platform.security.request_limits import RequestSizeLimiter
from jarvis_brain.service_paths import SERVICE_ROOT


SECRET_ENV_KEYWORDS = ("KEY", "TOKEN", "SECRET", "PASSWORD", "DATABASE_URL")


class StartupChecks:
    """Safe startup/configuration checks for deployability."""

    def __init__(
        self,
        auth_guard: AuthGuard | None = None,
        request_size_limiter: RequestSizeLimiter | None = None,
    ) -> None:
        """Create startup checks."""
        self.auth_guard = auth_guard or AuthGuard()
        self.request_size_limiter = request_size_limiter or RequestSizeLimiter()

    def run(self) -> dict[str, Any]:
        """Return startup health and config checks without secrets."""
        auth = self.auth_guard.status()
        # Resolved per call against Brain's own repository. The module-level
        # `ENV_PATH` this used to read is fixed at import of the shared config
        # package, before any service exists, and pointed at Core in the
        # composed workspace — so Brain reported its `.env` missing while
        # running on it.
        env_file = env_path_for(SERVICE_ROOT)
        checks = {
            "environment_file_present": env_file.exists(),
            "auth_enabled": auth.enabled,
            "request_size_limit_enabled": True,
            "request_size_limit_bytes": self.request_size_limiter.max_bytes,
            "secrets_redacted": True,
            "database_url_configured": bool(os.getenv("DATABASE_URL")),
            "test_database_url_configured": bool(os.getenv("TEST_DATABASE_URL")),
        }
        warnings: list[str] = []
        if not auth.enabled:
            warnings.append("Admin auth is disabled for development.")
        if not env_file.exists():
            warnings.append("Project .env file was not found.")
        return {
            "status": "ok",
            "checks": checks,
            "warnings": warnings,
        }

    def redacted_config(self) -> dict[str, Any]:
        """Return a safe view of selected configuration."""
        keys = [
            "LLM_ENABLED",
            "LLM_PROVIDER",
            "VOICE_TTS_PROVIDER",
            "VOICE_LOCAL_TTS_ENABLED",
            "JARVIS_AUTH_ENABLED",
            "JARVIS_MAX_REQUEST_BYTES",
            "DATABASE_URL",
            "TEST_DATABASE_URL",
            "JARVIS_ADMIN_API_KEY",
        ]
        values: dict[str, str | None] = {}
        for key in keys:
            value = os.getenv(key)
            values[key] = self._redact_value(key, value)
        return {
            "status": "ok",
            "config": values,
            "secrets_redacted": True,
        }

    def security_check(self) -> dict[str, Any]:
        """Return safe production-hardening security status."""
        auth = self.auth_guard.status()
        return {
            "status": "ok",
            "auth": auth.__dict__,
            "request_limits": self.request_size_limiter.status(),
            "rate_limiting": {
                "enabled": True,
                "scope": "high-risk endpoints",
            },
            "security_headers": {
                "enabled": True,
                "headers": [
                    "X-Content-Type-Options",
                    "X-Frame-Options",
                    "Referrer-Policy",
                    "Content-Security-Policy",
                ],
            },
            "remote_execution_enabled": False,
            "secrets_exposed": False,
        }

    def dependency_health(self) -> dict[str, Any]:
        """Return lightweight dependency health checks."""
        return {
            "status": "ok",
            "dependencies": {
                "database_configured": bool(os.getenv("DATABASE_URL")),
                "llm_provider": os.getenv("LLM_PROVIDER", "mock"),
                "voice_tts_provider": os.getenv("VOICE_TTS_PROVIDER", "mock"),
            },
        }

    def _redact_value(self, key: str, value: str | None) -> str | None:
        if value is None:
            return None
        if any(secret_key in key.upper() for secret_key in SECRET_ENV_KEYWORDS):
            return "[REDACTED]" if value else ""
        return value
