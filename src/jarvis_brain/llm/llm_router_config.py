import os

from jarvis_platform.config import load_app_environment
from jarvis_brain.service_paths import SERVICE_ROOT
from jarvis_platform.schemas.llm import LLMProviderName


class LLMRouterConfig:
    """Environment-backed router configuration with safe defaults."""

    def __init__(self) -> None:
        load_app_environment(SERVICE_ROOT)
        self.default_provider = _provider_from_env("LLM_DEFAULT_PROVIDER", LLMProviderName.OLLAMA)
        self.enabled = _env_truthy(os.getenv("LLM_ROUTER_ENABLED", "true"))
        self.retry_count = _int_from_env("LLM_ROUTER_RETRY_COUNT", 1)
        self.timeout_seconds = _int_from_env("LLM_TIMEOUT_SECONDS", 60)
        self.latency_budget_ms = _int_from_env("LLM_ROUTER_LATENCY_BUDGET_MS", 12000)
        self.default_privacy = os.getenv("LLM_PRIVACY_DEFAULT", "local_preferred")
        self.preferred_providers = _providers_from_env(
            "LLM_PREFERRED_PROVIDERS",
            [self.default_provider, LLMProviderName.MOCK],
        )
        self.coding_chain = _providers_from_env(
            "LLM_ROUTING_CODING",
            [LLMProviderName.OLLAMA, LLMProviderName.MOCK],
        )
        self.reasoning_chain = _providers_from_env(
            "LLM_ROUTING_REASONING",
            [LLMProviderName.OLLAMA, LLMProviderName.MOCK],
        )
        self.simple_chain = _providers_from_env(
            "LLM_ROUTING_SIMPLE",
            [LLMProviderName.OLLAMA, LLMProviderName.MOCK],
        )


def _providers_from_env(name: str, default: list[LLMProviderName]) -> list[LLMProviderName]:
    value = os.getenv(name)
    if not value:
        return default
    providers: list[LLMProviderName] = []
    for item in value.split(","):
        providers.append(_provider_from_text(item.strip()))
    return providers or default


def _provider_from_env(name: str, default: LLMProviderName) -> LLMProviderName:
    return _provider_from_text(os.getenv(name, default.value))


def _provider_from_text(value: str | None) -> LLMProviderName:
    try:
        return LLMProviderName((value or "").strip().lower())
    except ValueError:
        return LLMProviderName.UNKNOWN


def _env_truthy(value: str | None) -> bool:
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value >= 0 else default
