from collections.abc import Callable

from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_brain.llm.llm_provider_factory import create_llm_provider
from jarvis_brain.llm.mock_llm_provider import MockLLMProvider
from jarvis_brain.llm.ollama_provider import OllamaProvider
from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.provider import OmniRouteGatewayProvider
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import OmniRouteRoute, RouteHealthState
from jarvis_platform.schemas.llm import LLMProviderName
from jarvis_platform.schemas.llm_router import LLMProviderHealth


class LLMProviderRegistry:
    """Registry for configured and future providers."""

    future_providers = {
        LLMProviderName.OPENAI,
        LLMProviderName.ANTHROPIC,
        LLMProviderName.GEMINI,
        LLMProviderName.AIRLLM,
        LLMProviderName.LLAMA_CPP,
        LLMProviderName.LM_STUDIO,
        LLMProviderName.VLLM,
    }

    def __init__(
        self,
        provider_factory: Callable[[str | None], LLMProvider] | None = None,
        omniroute_settings: OmniRouteSettings | None = None,
        omniroute_registry: OmniRouteRouteRegistry | None = None,
    ) -> None:
        self.provider_factory = provider_factory or create_llm_provider
        self.omniroute_settings = omniroute_settings or OmniRouteSettings()
        self.omniroute_registry = omniroute_registry or OmniRouteRouteRegistry(
            self.omniroute_settings
        )

    def create_provider(self, provider_name: LLMProviderName, model: str) -> LLMProvider | None:
        if provider_name == LLMProviderName.MOCK:
            return MockLLMProvider()
        if provider_name == LLMProviderName.OLLAMA:
            provider = self.provider_factory(model)
            if provider.name == LLMProviderName.MOCK and not isinstance(provider, MockLLMProvider):
                return provider
            return provider
        if provider_name == LLMProviderName.OMNIROUTE:
            route = self.omniroute_registry.find_model(model)
            if route is None or not self.omniroute_registry.is_operator_approved(route):
                return None
            return self.create_omniroute_provider(route)
        if provider_name in self.future_providers:
            return None
        return None

    def create_omniroute_provider(
        self, route: OmniRouteRoute
    ) -> OmniRouteGatewayProvider:
        return OmniRouteGatewayProvider(route, self.omniroute_settings)

    def health(self, provider_name: LLMProviderName, models: list[str] | None = None) -> LLMProviderHealth:
        if provider_name == LLMProviderName.OMNIROUTE:
            approved = self.omniroute_registry.approved_routes()
            return LLMProviderHealth(
                provider=provider_name,
                enabled=self.omniroute_settings.enabled,
                available=self.omniroute_settings.ready_for_requests and bool(approved),
                models=[route.model_id for route in approved],
                metadata={
                    "state": (
                        RouteHealthState.STARTING.value
                        if self.omniroute_settings.ready_for_requests and approved
                        else RouteHealthState.DISABLED.value
                        if not self.omniroute_settings.enabled
                        else RouteHealthState.MISCONFIGURED.value
                    ),
                    **self.omniroute_settings.safe_status(),
                    **self.omniroute_registry.safe_summary(),
                    "availability_semantics": "configured_not_live_probed",
                },
            )
        provider = self.create_provider(provider_name, models[0] if models else None) if models else self.create_provider(provider_name, None)
        if provider is None:
            return LLMProviderHealth(provider=provider_name, enabled=False, available=False, models=models or [], metadata={"adapter": "planned"})
        return LLMProviderHealth(
            provider=provider.name,
            enabled=not isinstance(provider, OllamaProvider) or provider.enabled,
            available=provider.is_available(),
            models=models or [provider.model],
        )

    def list_provider_names(self) -> list[LLMProviderName]:
        return [
            LLMProviderName.MOCK,
            LLMProviderName.OLLAMA,
            LLMProviderName.OPENAI,
            LLMProviderName.ANTHROPIC,
            LLMProviderName.GEMINI,
            LLMProviderName.AIRLLM,
            LLMProviderName.LLAMA_CPP,
            LLMProviderName.LM_STUDIO,
            LLMProviderName.VLLM,
            LLMProviderName.OMNIROUTE,
        ]
