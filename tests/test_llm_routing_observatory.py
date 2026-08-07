import pytest
from types import SimpleNamespace

from fastapi.testclient import TestClient

from jarvis_brain.llm.model_registry import LLMModelRegistry
from jarvis_brain.llm.omniroute.registry import OmniRouteRouteRegistry
from jarvis_brain.llm.omniroute.schemas import (
    ModelCapability,
    OmniRouteRoute,
    RouteLocality,
    RoutePrivacy,
    TaskCategory,
)
from jarvis_brain.llm.routing_observatory import LLMRoutingObservatory
from jarvis_brain.llm.topology_identity import (
    model_topology_id,
    provider_topology_id,
    route_topology_id,
)
from jarvis_platform.adapters.builtin import IntelligenceRouterAdapter
from jarvis_platform.adapters.enums import AdapterPrivacy
from jarvis_platform.adapters.schemas import AdapterExecutionContext, AdapterRequest
from jarvis_platform.adapters.selector import AdapterSelector
from jarvis_platform.adapters.registry import AdapterRegistry
from jarvis_platform.adapters.capabilities import LLM_GENERATE
from jarvis_brain.app import app
from jarvis_platform.observability.event_logger import EventLogger
from jarvis_platform.schemas.llm import LLMProviderName, LLMTaskType
from jarvis_platform.schemas.llm_router import (

    LLMPrivacyClass,
    LLMProviderHealth,
    LLMRoutingDecision,
)


@pytest.fixture(autouse=True)
def pinned_llm_models(monkeypatch):
    """Pin the model identities these tests assert on.

    The registry reads `LLM_GENERAL_MODEL` / `LLM_CODING_MODEL` from the
    environment, so a developer `.env` that points both at one model made these
    assertions describe that machine rather than the behaviour under test. The
    expectation is established here instead of being inherited.
    """
    monkeypatch.setenv("LLM_GENERAL_MODEL", "llama3.1:8b")
    monkeypatch.setenv("LLM_CODING_MODEL", "qwen2.5-coder:7b")
    yield


class FakeSettings:
    enabled = False
    api_key = ""
    allow_cloud = False
    ready_for_requests = False
    allowed_routes = frozenset()
    allowed_providers = frozenset()
    allowed_models = frozenset()


class ApprovedSettings(FakeSettings):
    enabled = True
    api_key = "test-only"
    ready_for_requests = True


class FakeRouter:
    def __init__(
        self,
        routes: list[OmniRouteRoute] | None = None,
        *,
        settings: FakeSettings | None = None,
    ) -> None:
        self.config = SimpleNamespace(enabled=True)
        self.model_registry = LLMModelRegistry()
        self.omniroute_settings = settings or FakeSettings()
        self.omniroute_registry = OmniRouteRouteRegistry(
            self.omniroute_settings, routes=routes or []
        )

    @staticmethod
    def list_provider_health() -> list[LLMProviderHealth]:
        return [
            LLMProviderHealth(
                provider=LLMProviderName.OLLAMA,
                enabled=True,
                available=True,
                models=["local-model"],
            )
        ]


def _route(
    provider_id: str = "nvidia",
    *,
    enabled: bool = False,
    model_id: str | None = None,
    locality: RouteLocality = RouteLocality.CLOUD,
    terms_status: str = "unreviewed",
) -> OmniRouteRoute:
    return OmniRouteRoute(
        route_id=f"{provider_id}-route",
        provider_id=provider_id,
        model_id=model_id or f"{provider_id}/model",
        display_name=f"{provider_id.title()} model",
        locality=locality,
        privacy_class=(
            RoutePrivacy.LOCAL_ONLY
            if locality == RouteLocality.LOCAL
            else RoutePrivacy.CLOUD_ALLOWED
        ),
        capabilities=[ModelCapability.REASONING],
        task_categories=[TaskCategory.COMPLEX_REASONING],
        context_window=8192,
        enabled=enabled,
        production_approved=enabled and terms_status == "approved",
        terms_status=terms_status,
    )


def _decision(request_id: str = "request-1") -> LLMRoutingDecision:
    del request_id
    return LLMRoutingDecision(
        decision_id="decision-1",
        task_type=LLMTaskType.REASONING,
        privacy_class=LLMPrivacyClass.LOCAL_ONLY,
        complexity_score=4,
        selected_provider=LLMProviderName.OLLAMA,
        selected_model="local-model",
        reason="Local reasoning route selected.",
    )


def _observatory() -> tuple[LLMRoutingObservatory, EventLogger]:
    logger = EventLogger(max_events=100, persist_event=lambda event: None)
    return LLMRoutingObservatory(logger), logger


def test_catalog_snapshot_distinguishes_available_and_policy_blocked() -> None:
    observatory, _ = _observatory()
    snapshot = observatory.snapshot(FakeRouter([_route()]))
    providers = {item.topology_node_id: item for item in snapshot.providers}
    assert providers["direct:ollama"].available is True
    assert providers["direct:ollama"].locality == "local"
    assert providers["omniroute:nvidia"].health.value == "disabled"
    assert providers["omniroute:nvidia"].policy_status == "route_disabled"
    assert providers["omniroute:nvidia"].models[0].block_reason == "ROUTE_DISABLED"
    serialized = snapshot.model_dump_json()
    assert "api_key" not in serialized
    assert '"messages"' not in serialized
    assert '"content"' not in serialized


def test_catalog_diff_reports_addition_and_removal_without_treating_health_as_removal() -> None:
    observatory, logger = _observatory()
    router = FakeRouter([_route("nvidia")])
    assert observatory.snapshot(router).catalog_diff is None
    router.omniroute_registry = OmniRouteRouteRegistry(
        router.omniroute_settings, routes=[_route("groq")]
    )
    diff = observatory.snapshot(router).catalog_diff
    assert diff is not None
    assert diff.added_provider_ids == ["groq"]
    assert diff.removed_provider_ids == ["nvidia"]
    assert diff.added_topology_node_ids == ["omniroute:groq"]
    assert diff.removed_topology_node_ids == ["omniroute:nvidia"]
    event_types = [item["metadata"]["event_type"] for item in logger.list_events()]
    assert "llm.provider.added" in event_types
    assert "llm.provider.removed" in event_types


def test_route_lifecycle_is_bounded_and_contains_no_content() -> None:
    observatory, logger = _observatory()
    observatory.request_received("request-1", source="voice")
    observatory.classified(
        "request-1",
        task_category="complex_reasoning",
        privacy_class="local_only",
        complexity=4,
        required_capabilities=["reasoning"],
    )
    observatory.route_selected(
        _decision(),
        request_id="request-1",
        source="voice",
        gateway="direct",
        provider_id="ollama",
        reason_codes=["TASK_MATCH", "PRIVACY_ALLOWED"],
    )
    observatory.connecting("request-1")
    observatory.stream_delta("request-1", 1)
    observatory.stream_delta("request-1", 2)
    observatory.validating("request-1")
    observatory.completed("request-1")
    snapshot = observatory.snapshot(FakeRouter())
    assert snapshot.active_route is not None
    assert snapshot.active_route.state.value == "completed"
    assert snapshot.active_route.chunk_count == 2
    assert len(snapshot.recent_route_changes) == 1
    serialized = str(logger.list_events()).lower()
    assert "prompt" not in serialized
    assert "response_text" not in serialized
    assert "llm.stream.delta" in serialized


def test_failure_after_visible_output_disallows_fallback() -> None:
    observatory, logger = _observatory()
    observatory.route_selected(
        _decision(),
        request_id="request-1",
        source="api",
        gateway="direct",
        provider_id="ollama",
    )
    observatory.stream_delta("request-1", 1)
    observatory.failed(
        "request-1", safe_reason="STREAM_FAILED_AFTER_OUTPUT", fallback_allowed=False
    )
    failure = next(
        item
        for item in logger.list_events()
        if item["metadata"]["event_type"] == "llm.route.failed"
    )
    assert failure["metadata"]["fallback_allowed"] is False


def test_topology_endpoint_is_owned_and_safe() -> None:
    response = TestClient(app).get(
        "/llm/router/topology", headers={"X-Jarvis-User-Id": "local-user"}
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["safety"]["observational_only"] is True
    assert payload["safety"]["credentials_included"] is False
    assert len(payload["providers"]) <= 24
    assert all("topology_node_id" in item for item in payload["providers"])


def test_topology_endpoint_rejects_anonymous_access() -> None:
    assert TestClient(app).get("/llm/router/topology").status_code == 401


def test_intelligence_router_remains_eligible_for_local_only_requests() -> None:
    registry = AdapterRegistry()
    registry.register(IntelligenceRouterAdapter(SimpleNamespace(list_provider_health=lambda: [])))
    adapter, _ = AdapterSelector().select(
        registry,
        AdapterRequest(capability=LLM_GENERATE, privacy_requirement=AdapterPrivacy.LOCAL),
        AdapterExecutionContext(privacy_policy=AdapterPrivacy.LOCAL),
    )
    assert adapter.adapter_id == "llm-router"


def test_stream_endpoint_uses_authoritative_router(monkeypatch) -> None:
    observed: list[str] = []

    def fake_stream(request):
        observed.append(request.request_id)
        yield {"type": "token", "content": "Routing online."}
        yield {"type": "done", "metadata": {"provider": "ollama"}}

    monkeypatch.setattr(
        "jarvis_brain.routes.llm.intelligence_router.generate_stream", fake_stream
    )
    response = TestClient(app).post(
        "/llm/generate/stream",
        json={"messages": [{"role": "user", "content": "status"}]},
    )
    assert response.status_code == 200
    assert observed
    assert "Routing online." in response.text


def test_topology_identity_helpers_are_transport_aware_and_escape_segments() -> None:
    assert provider_topology_id(gateway="direct", provider_id="ollama") == "direct:ollama"
    assert (
        provider_topology_id(gateway="omniroute", provider_id="ollama")
        == "omniroute:ollama"
    )
    assert model_topology_id(
        gateway="direct", provider_id="ollama", model_id="llama3.2:3b"
    ) == "direct:ollama:llama3.2%3A3b"
    assert route_topology_id(
        gateway="omniroute",
        provider_id="ollama",
        model_id="ollama/llama3.1:8b",
        route_id="local-reviewed-placeholder",
    ) == (
        "omniroute:ollama:ollama%2Fllama3.1%3A8b:"
        "local-reviewed-placeholder"
    )


def test_direct_and_disabled_omniroute_ollama_are_separate_nodes() -> None:
    route = _route(
        "ollama",
        model_id="ollama/llama3.1:8b",
        locality=RouteLocality.LOCAL,
    )
    snapshot = _observatory()[0].snapshot(FakeRouter([route]))
    providers = {item.topology_node_id: item for item in snapshot.providers}

    assert set(providers) == {"direct:ollama", "omniroute:ollama"}
    direct = providers["direct:ollama"]
    gateway = providers["omniroute:ollama"]
    assert direct.provider_id == gateway.provider_id == "ollama"
    assert direct.gateway == "direct"
    assert gateway.gateway == "omniroute"
    assert direct.available is True
    assert gateway.health.value == "disabled"
    assert gateway.models[0].model_id == "ollama/llama3.1:8b"
    assert gateway.models[0].block_reason == "ROUTE_DISABLED"
    assert all(model.gateway == "direct" for model in direct.models)
    assert all(model.gateway == "omniroute" for model in gateway.models)


def test_approved_omniroute_ollama_remains_separate_from_direct_fallback() -> None:
    route = _route(
        "ollama",
        enabled=True,
        model_id="ollama/llama3.1:8b",
        locality=RouteLocality.LOCAL,
        terms_status="approved",
    )
    snapshot = _observatory()[0].snapshot(
        FakeRouter([route], settings=ApprovedSettings())
    )
    providers = {item.topology_node_id: item for item in snapshot.providers}

    assert set(providers) == {"direct:ollama", "omniroute:ollama"}
    assert providers["omniroute:ollama"].approved is True
    assert providers["omniroute:ollama"].available is True
    assert providers["direct:ollama"].available is True
    assert (
        providers["omniroute:ollama"].models[0].topology_model_id
        != providers["direct:ollama"].models[0].topology_model_id
    )


def test_route_lifecycle_selects_only_the_matching_transport() -> None:
    route = _route(
        "ollama",
        enabled=True,
        model_id="ollama/llama3.1:8b",
        locality=RouteLocality.LOCAL,
        terms_status="approved",
    )
    router = FakeRouter([route], settings=ApprovedSettings())
    observatory, _ = _observatory()
    gateway_decision = _decision().model_copy(
        update={"selected_model": "ollama/llama3.1:8b"}
    )
    observatory.route_selected(
        gateway_decision,
        request_id="request-gateway",
        source="voice",
        gateway="omniroute",
        provider_id="ollama",
        route_id="ollama-route",
    )
    gateway_snapshot = observatory.snapshot(router)
    selected = {
        item.topology_node_id: item.selected for item in gateway_snapshot.providers
    }
    assert selected == {"direct:ollama": False, "omniroute:ollama": True}

    observatory.failed(
        "request-gateway",
        safe_reason="PROVIDER_UNAVAILABLE",
        fallback_allowed=True,
    )
    observatory.route_selected(
        _decision(),
        request_id="request-gateway",
        source="voice",
        gateway="direct",
        provider_id="ollama",
        fallback_from="ollama",
        fallback_reason="PRIMARY_FAILED_BEFORE_OUTPUT",
    )
    fallback_snapshot = observatory.snapshot(router)
    selected = {
        item.topology_node_id: item.selected for item in fallback_snapshot.providers
    }
    assert selected == {"direct:ollama": True, "omniroute:ollama": False}
    assert fallback_snapshot.active_route is not None
    assert fallback_snapshot.active_route.provider_id == "ollama"
    assert fallback_snapshot.active_route.topology_node_id == "direct:ollama"


def test_catalog_reconciliation_preserves_stable_transport_nodes() -> None:
    route = _route(
        "ollama",
        model_id="ollama/llama3.1:8b",
        locality=RouteLocality.LOCAL,
    )
    router = FakeRouter([route])
    observatory, _ = _observatory()
    first = observatory.snapshot(router)
    second = observatory.snapshot(router)

    assert first.catalog_diff is None
    assert second.catalog_diff is None
    assert [item.topology_node_id for item in first.providers] == [
        item.topology_node_id for item in second.providers
    ]
    assert len({item.topology_node_id for item in second.providers}) == 2
    assert len(
        {
            model.topology_model_id
            for provider in second.providers
            for model in provider.models
        }
    ) == sum(len(provider.models) for provider in second.providers)


def test_same_model_id_on_two_transports_has_unique_model_and_edge_identities(monkeypatch) -> None:
    # This test is about one model id appearing on two transports, so the direct
    # transport has to actually carry that id.
    monkeypatch.setenv("LLM_GENERAL_MODEL", "llama3.2:3b")
    route = _route(
        "ollama",
        model_id="llama3.2:3b",
        locality=RouteLocality.LOCAL,
    )
    snapshot = _observatory()[0].snapshot(FakeRouter([route]))
    providers = {item.topology_node_id: item for item in snapshot.providers}
    direct_model = next(
        item
        for item in providers["direct:ollama"].models
        if item.model_id == "llama3.2:3b"
    )
    gateway_model = providers["omniroute:ollama"].models[0]

    assert direct_model.model_id == gateway_model.model_id == "llama3.2:3b"
    assert direct_model.topology_model_id != gateway_model.topology_model_id
    assert direct_model.topology_route_id != gateway_model.topology_route_id
    assert direct_model.provider_id == gateway_model.provider_id == "ollama"
    assert {direct_model.gateway, gateway_model.gateway} == {"direct", "omniroute"}
