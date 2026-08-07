from jarvis_brain.llm.llm_model_router import LLMModelRouter
from jarvis_brain.llm.llm_provider_factory import create_llm_provider, create_model_router
from jarvis_brain.llm.mock_llm_provider import MockLLMProvider
from jarvis_brain.llm.ollama_provider import OllamaProvider


def test_disabled_env_returns_mock_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "false")

    assert isinstance(create_llm_provider(), MockLLMProvider)


def test_ollama_enabled_returns_ollama_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    assert isinstance(create_llm_provider(), OllamaProvider)


def test_custom_model_override_is_honored(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "ollama")

    provider = create_llm_provider(model="custom-model")

    assert provider.model == "custom-model"


def test_enabled_without_provider_defaults_to_ollama(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "")

    assert isinstance(create_llm_provider(), OllamaProvider)


def test_unknown_provider_returns_mock(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "true")
    monkeypatch.setenv("LLM_PROVIDER", "other")

    assert isinstance(create_llm_provider(), MockLLMProvider)


def test_missing_env_does_not_crash(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "false")
    monkeypatch.delenv("LLM_PROVIDER", raising=False)

    assert create_llm_provider().model == "mock-model"


def test_create_model_router_reads_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GENERAL_MODEL", "general")
    monkeypatch.setenv("LLM_CODING_MODEL", "coding")

    router = create_model_router()

    assert isinstance(router, LLMModelRouter)
    assert router.general_model == "general"
    assert router.coding_model == "coding"
