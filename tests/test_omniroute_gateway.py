import json
import urllib.error

import pytest

from jarvis_brain.llm.omniroute.classifier import IntelligentTaskClassifier
from jarvis_brain.llm.omniroute.config import OmniRouteConfigurationError, OmniRouteSettings
from jarvis_brain.llm.omniroute.discovery import OmniRouteDiscoveryClient
from jarvis_brain.llm.omniroute.policy import OmniRouteSelectionPolicy
from jarvis_brain.llm.omniroute.provider import OmniRouteGatewayProvider, _RejectRedirects
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import (
    ModelCapability,
    OmniRouteRoute,
    RouteLocality,
    RoutePrivacy,
    TaskCategory,
)
from jarvis_brain.llm.intelligence_router import IntelligenceRouter
from jarvis_brain.llm.router_telemetry import LLMRouterTelemetry
from jarvis_platform.schemas.llm import (
    LLMMessage,
    LLMProviderName,
    LLMRequest,
    LLMRole,
    LLMStatus,
)


class FakeResponse:
    def __init__(self, body=b"", *, chunks=None, status=200) -> None:
        self.body = body
        self.chunks = chunks or []
        self.status = status
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.closed = True
        return None

    def read(self, _size=None):
        return self.body

    def __iter__(self):
        return iter(self.chunks)


class FakeOpener:
    def __init__(self, response=None, error=None) -> None:
        self.response = response
        self.error = error
        self.request = None

    def open(self, request, timeout):
        self.request = request
        if self.error:
            raise self.error
        return self.response


def _settings(monkeypatch, **values) -> OmniRouteSettings:
    defaults = {
        "OMNIROUTE_ENABLED": "true",
        "OMNIROUTE_BASE_URL": "http://127.0.0.1:20128/v1",
        "OMNIROUTE_API_KEY": "test-only-key",
        "OMNIROUTE_ALLOW_CLOUD": "false",
        "OMNIROUTE_ALLOWED_PROVIDERS": "",
        "OMNIROUTE_ALLOWED_MODELS": "",
        "OMNIROUTE_ALLOWED_ROUTES": "",
    }
    defaults.update(values)
    for key, value in defaults.items():
        monkeypatch.setenv(key, value)
    return OmniRouteSettings()


def _route(**updates) -> OmniRouteRoute:
    values = {
        "route_id": "nvidia-reasoning",
        "provider_id": "nvidia",
        "model_id": "nvidia/reasoning-model",
        "display_name": "Reviewed reasoning route",
        "locality": RouteLocality.CLOUD,
        "privacy_class": RoutePrivacy.CLOUD_ALLOWED,
        "capabilities": [
            ModelCapability.REASONING,
            ModelCapability.CODING,
            ModelCapability.LONG_CONTEXT,
            ModelCapability.HIGH_RELIABILITY,
        ],
        "task_categories": [TaskCategory.SOFTWARE_ARCHITECTURE],
        "context_window": 32768,
        "terms_status": "approved",
        "development_only": True,
        "enabled": True,
        "expected_quality_class": "high",
        "expected_latency_class": "balanced",
    }
    values.update(updates)
    return OmniRouteRoute(**values)


def _request() -> LLMRequest:
    return LLMRequest(
        request_id="request-1",
        messages=[LLMMessage(role=LLMRole.USER, content="Review this architecture")],
    )


def test_omniroute_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("OMNIROUTE_ENABLED", raising=False)
    monkeypatch.delenv("OMNIROUTE_API_KEY", raising=False)
    settings = OmniRouteSettings()
    assert settings.enabled is False
    assert settings.ready_for_requests is False
    assert settings.allow_cloud is False
    assert settings.allow_combos is False
    assert settings.allow_internal_fallback is False
    assert settings.allow_compression is False


@pytest.mark.parametrize(
    "url",
    [
        "https://127.0.0.1:20128/v1",
        "http://192.168.1.20:20128/v1",
        "http://gateway.example/v1",
        "http://user:pass@127.0.0.1:20128/v1",
        "http://127.0.0.1:20128/v1?token=secret",
        "http://127.0.0.1:20128/api",
    ],
)
def test_remote_or_ambiguous_gateway_url_is_rejected(monkeypatch, url) -> None:
    monkeypatch.setenv("OMNIROUTE_BASE_URL", url)
    with pytest.raises(OmniRouteConfigurationError):
        OmniRouteSettings()


def test_cloud_route_requires_operator_and_request_authorization(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_ALLOW_CLOUD="true")
    registry = OmniRouteRouteRegistry(settings, routes=[_route()])
    classifier = IntelligentTaskClassifier()
    policy = OmniRouteSelectionPolicy(registry, settings)
    messages = [
        LLMMessage(
            role=LLMRole.USER,
            content="Design a complex software architecture and compare tradeoffs",
        )
    ]

    unauthorized = classifier.classify(
        messages, {"privacy_class": "cloud_allowed"}
    )
    authorized = classifier.classify(
        messages,
        {"privacy_class": "cloud_allowed", "_cloud_authorized": True},
    )

    assert policy.select(unauthorized).selected is False
    assert policy.select(authorized).selected is True


def test_local_only_and_unknown_locality_fail_closed(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_ALLOW_CLOUD="true")
    routes = [_route(), _route(route_id="unknown", locality="unknown")]
    registry = OmniRouteRouteRegistry(settings, routes=routes)
    classification = IntelligentTaskClassifier().classify(
        [LLMMessage(role=LLMRole.USER, content="My API key is private; analyze architecture")],
        {},
    )
    decision = OmniRouteSelectionPolicy(registry, settings).select(classification)
    assert decision.selected is False
    assert "local_only_violation" in decision.rejected_routes["nvidia-reasoning"]
    assert "unknown_locality" in decision.rejected_routes["unknown"]


def test_discovery_normalizes_deduplicates_and_does_not_enable(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    registry = OmniRouteRouteRegistry(
        settings, routes=[_route(enabled=False, terms_status="unreviewed")]
    )
    client = OmniRouteDiscoveryClient(settings, registry)
    payload = {
        "data": [
            {"id": "nvidia/reasoning-model", "owned_by": "nvidia"},
            {"id": "nvidia/reasoning-model", "owned_by": "nvidia"},
            {"id": "unknown/model", "owned_by": "unknown"},
        ]
    }
    client._opener = FakeOpener(FakeResponse(json.dumps(payload).encode()))
    snapshot = client.fetch()
    assert snapshot.state.value == "ready"
    assert snapshot.duplicate_count == 1
    assert registry.approved_routes() == []
    assert "nvidia/reasoning-model" in snapshot.discovered_unapproved


def test_discovery_sanitizes_unauthorized_error(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    client = OmniRouteDiscoveryClient(settings)
    client._opener = FakeOpener(
        error=urllib.error.HTTPError(
            settings.base_url, 401, "contains secret body", {}, None
        )
    )
    snapshot = client.fetch()
    assert snapshot.state.value == "unauthorized"
    assert "secret" not in (snapshot.safe_error or "")


def test_redirect_handler_rejects_every_redirect() -> None:
    handler = _RejectRedirects()
    request = type("Request", (), {"full_url": "http://127.0.0.1:20128/v1/models"})()
    with pytest.raises(urllib.error.HTTPError):
        handler.redirect_request(
            request, None, 302, "redirect", {}, "https://remote.example/v1/models"
        )


def test_provider_uses_exact_provider_route_and_bearer_auth(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_ALLOW_CLOUD="true")
    provider = OmniRouteGatewayProvider(_route(), settings)
    response = FakeResponse(
        json.dumps({"choices": [{"message": {"content": "Safe result"}}]}).encode()
    )
    opener = FakeOpener(response)
    provider._opener = opener
    result = provider.generate(_request())
    assert result.status == LLMStatus.SUCCESS
    assert opener.request.full_url.endswith(
        "/v1/providers/nvidia/chat/completions"
    )
    assert opener.request.headers["Authorization"] == "Bearer test-only-key"
    assert result.raw_metadata["route_id"] == "nvidia-reasoning"
    assert "endpoint" not in result.raw_metadata


def test_provider_stream_handles_split_utf8_done_and_bounds(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_MAX_RESPONSE_BYTES="1024")
    provider = OmniRouteGatewayProvider(_route(), settings)
    encoded = 'data: {"choices":[{"delta":{"content":"cafe ☕"}}]}\n'.encode()
    provider._opener = FakeOpener(
        FakeResponse(chunks=[encoded[:-2], encoded[-2:], b"data: [DONE]\n"])
    )
    events = list(provider.generate_stream(_request()))
    assert [item["content"] for item in events if item["type"] == "token"] == [
        "cafe ☕"
    ]
    assert events[-1]["type"] == "done"
    assert events[-1]["metadata"]["route_id"] == "nvidia-reasoning"
    assert "endpoint" not in events[-1]["metadata"]


def test_registry_rejects_duplicate_route_ids(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    with pytest.raises(ValueError):
        OmniRouteRouteRegistry(settings, routes=[_route(), _route()])


def test_registry_rejects_cross_privacy_fallback(monkeypatch) -> None:
    settings = _settings(monkeypatch)
    local = _route(
        route_id="local",
        provider_id="ollama",
        model_id="ollama/local",
        locality="local",
        privacy_class="local_only",
        fallback_route_ids=["nvidia-reasoning"],
    )
    with pytest.raises(ValueError):
        OmniRouteRouteRegistry(settings, routes=[local, _route()])


def test_intelligence_router_selects_gateway_only_with_explicit_permission(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_ALLOW_CLOUD="true")
    registry = OmniRouteRouteRegistry(settings, routes=[_route()])
    router = IntelligenceRouter(
        omniroute_settings=settings,
        omniroute_registry=registry,
        omniroute_policy=OmniRouteSelectionPolicy(registry, settings),
        telemetry=LLMRouterTelemetry(),
    )
    messages = [
        LLMMessage(
            role=LLMRole.USER,
            content="Design a complex software architecture and compare tradeoffs",
        )
    ]
    local = router.route_only(messages, {"privacy_class": "cloud_allowed"})
    cloud = router.route_only(
        messages,
        {"privacy_class": "cloud_allowed", "_cloud_authorized": True},
    )
    assert local.selected_provider == LLMProviderName.OLLAMA
    assert cloud.selected_provider == LLMProviderName.OMNIROUTE
    assert cloud.metadata["selected_route_id"] == "nvidia-reasoning"


class _StreamProvider:
    def __init__(self, name, events) -> None:
        self.name = name
        self.model = f"{name.value}-model"
        self.events = events
        self.calls = 0

    def is_available(self):
        return True

    def generate_stream(self, request):
        self.calls += 1
        yield from self.events


class _StreamRegistry:
    def __init__(self, settings, registry, gateway, local) -> None:
        self.omniroute_settings = settings
        self.omniroute_registry = registry
        self.gateway = gateway
        self.local = local

    def create_omniroute_provider(self, route):
        return self.gateway

    def create_provider(self, provider_name, model):
        if provider_name == LLMProviderName.OLLAMA:
            return self.local
        return None

    def list_provider_names(self):
        return [LLMProviderName.OLLAMA, LLMProviderName.OMNIROUTE]

    def health(self, provider, models=None):
        raise AssertionError("Health is not used by this test.")


def _stream_router(monkeypatch, gateway_events):
    settings = _settings(monkeypatch, OMNIROUTE_ALLOW_CLOUD="true")
    registry = OmniRouteRouteRegistry(settings, routes=[_route()])
    gateway = _StreamProvider(LLMProviderName.OMNIROUTE, gateway_events)
    local = _StreamProvider(
        LLMProviderName.OLLAMA,
        [
            {"type": "token", "content": "local"},
            {"type": "done", "metadata": {}},
        ],
    )
    provider_registry = _StreamRegistry(settings, registry, gateway, local)
    router = IntelligenceRouter(
        provider_registry=provider_registry,
        omniroute_settings=settings,
        omniroute_registry=registry,
        omniroute_policy=OmniRouteSelectionPolicy(registry, settings),
        telemetry=LLMRouterTelemetry(),
    )
    request = LLMRequest(
        request_id="stream-1",
        messages=[
            LLMMessage(
                role=LLMRole.USER,
                content="Design a complex software architecture and compare tradeoffs",
            )
        ],
        metadata={"privacy_class": "cloud_allowed", "_cloud_authorized": True},
    )
    return router, request, gateway, local


def test_stream_falls_back_to_local_before_visible_output(monkeypatch) -> None:
    router, request, gateway, local = _stream_router(
        monkeypatch, [{"type": "error", "message": "upstream details"}]
    )
    events = list(router.generate_stream(request))
    assert [item.get("content") for item in events if item["type"] == "token"] == ["local"]
    assert gateway.calls == 1
    assert local.calls == 1
    assert events[-1]["metadata"]["fallback_used"] is True


def test_stream_never_restarts_after_visible_output(monkeypatch) -> None:
    router, request, gateway, local = _stream_router(
        monkeypatch,
        [
            {"type": "token", "content": "visible"},
            {"type": "error", "message": "upstream details"},
        ],
    )
    events = list(router.generate_stream(request))
    assert [item.get("content") for item in events if item["type"] == "token"] == ["visible"]
    assert events[-1]["metadata"]["fallback_blocked_after_visible_output"] is True
    assert gateway.calls == 1
    assert local.calls == 0


def test_oversized_provider_response_is_safely_rejected(monkeypatch) -> None:
    settings = _settings(monkeypatch, OMNIROUTE_MAX_RESPONSE_BYTES="1024")
    provider = OmniRouteGatewayProvider(_route(), settings)
    provider._opener = FakeOpener(FakeResponse(b"x" * 1025))
    result = provider.generate(_request())
    assert result.status == LLMStatus.ERROR
    assert "unexpected" in (result.error_message or "").lower()


def test_stream_cancellation_closes_http_response(monkeypatch) -> None:
    class CancelAfterFirstCheck:
        def __init__(self) -> None:
            self.checks = 0

        def raise_if_cancelled(self):
            self.checks += 1
            if self.checks > 1:
                raise RuntimeError("cancelled")

    settings = _settings(monkeypatch)
    provider = OmniRouteGatewayProvider(_route(), settings)
    response = FakeResponse(
        chunks=[
            b'data: {"choices":[{"delta":{"content":"first"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"second"}}]}\n',
        ]
    )
    provider._opener = FakeOpener(response)
    request = _request().model_copy(
        update={"metadata": {"_cancellation_token": CancelAfterFirstCheck()}}
    )
    with pytest.raises(RuntimeError, match="cancelled"):
        list(provider.generate_stream(request))
    assert response.closed is True
