import os

from jarvis_platform.config import load_app_environment
from jarvis_brain.service_paths import SERVICE_ROOT
from jarvis_brain.llm.llm_model_router import LLMModelRouter
from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_brain.llm.mock_llm_provider import MockLLMProvider
from jarvis_brain.llm.ollama_provider import OllamaProvider
from jarvis_brain.llm.openai_compatible_provider import OpenAICompatibleProvider


# OpenAI-compatible providers share one adapter; only the default base URL and
# the env var that carries the key differ. Add a hosted vendor here by name.
_OPENAI_COMPATIBLE_ENDPOINTS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "openai": "https://api.openai.com/v1",
    "groq": "https://api.groq.com/openai/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "together": "https://api.together.xyz/v1",
    "openai_compatible": "https://integrate.api.nvidia.com/v1",
}


def create_llm_provider(model: str | None = None) -> LLMProvider:
    """Create the configured LLM provider.

    The mock provider is the default so tests and local development do not
    require Ollama to be running.
    """
    load_app_environment(SERVICE_ROOT)
    if not _env_truthy(os.getenv("LLM_ENABLED")):
        return MockLLMProvider()

    provider_name = (os.getenv("LLM_PROVIDER") or "ollama").strip().lower()
    timeout = _int_from_env("LLM_TIMEOUT_SECONDS", 60)
    if provider_name == "ollama":
        return OllamaProvider(
            enabled=True,
            model=model
            or os.getenv("LLM_MODEL")
            or os.getenv("LLM_GENERAL_MODEL")
            or "llama3.1:8b",
            base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            timeout_seconds=timeout,
            keep_alive=os.getenv("OLLAMA_KEEP_ALIVE", "10m"),
        )

    if provider_name in _OPENAI_COMPATIBLE_ENDPOINTS:
        # LLM_BASE_URL overrides the vendor default so any OpenAI-compatible host
        # works. The key falls back across common vendor env var names.
        base_url = (
            os.getenv("LLM_BASE_URL")
            or _OPENAI_COMPATIBLE_ENDPOINTS[provider_name]
        )
        api_key = (
            os.getenv("LLM_API_KEY")
            or os.getenv("NVIDIA_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        )
        return OpenAICompatibleProvider(
            enabled=True,
            model=model
            or os.getenv("LLM_MODEL")
            or os.getenv("LLM_GENERAL_MODEL")
            or "meta/llama-3.1-8b-instruct",
            base_url=base_url,
            api_key=api_key,
            timeout_seconds=timeout,
        )

    return MockLLMProvider()


def create_model_router() -> LLMModelRouter:
    """Create an LLM model router from environment configuration."""
    load_app_environment(SERVICE_ROOT)
    return LLMModelRouter(
        general_model=os.getenv("LLM_GENERAL_MODEL", "llama3.1:8b"),
        coding_model=os.getenv("LLM_CODING_MODEL", "qwen2.5-coder:7b"),
        classification_model=os.getenv(
            "LLM_CLASSIFICATION_MODEL", "qwen2.5-coder:3b"
        ),
        default_model=os.getenv("LLM_MODEL", "llama3.1:8b"),
    )


def _env_truthy(value: str | None) -> bool:
    """Return True for common truthy environment values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _int_from_env(name: str, default: int) -> int:
    """Read a positive integer from an environment variable."""
    try:
        value = int(os.getenv(name, str(default)))
    except ValueError:
        return default
    return value if value > 0 else default
