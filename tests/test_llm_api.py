import pytest
from unittest.mock import patch

from fastapi.testclient import TestClient

from jarvis_brain.app import app
from jarvis_platform.observability.trace_service import observability_trace_service


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



client = TestClient(app)


def test_llm_status_returns_200() -> None:
    response = client.get("/llm/status")

    assert response.status_code == 200
    assert "provider" in response.json()


def test_llm_route_routes_fastapi_error_to_qwen() -> None:
    response = client.post("/llm/route", json={"text": "I have a FastAPI traceback"})

    assert response.status_code == 200
    assert response.json()["task_type"] == "coding"
    assert response.json()["model"] == "qwen2.5-coder:7b"


def test_llm_route_routes_general_message_to_configured_model(monkeypatch) -> None:
    monkeypatch.setenv("LLM_GENERAL_MODEL", "test-general-model")
    response = client.post("/llm/route", json={"text": "Tell me about Jarvis"})

    assert response.status_code == 200
    assert response.json()["task_type"] == "general"
    assert response.json()["model"] == "test-general-model"


def test_llm_generate_returns_mock_response_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "false")
    response = client.post(
        "/llm/generate",
        json={"messages": [{"role": "user", "content": "Say hello from Jarvis"}]},
    )

    assert response.status_code == 200
    assert "Mock LLM response" in response.json()["content"]


def test_llm_generate_redacts_secrets(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "false")
    response = client.post(
        "/llm/generate",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "my key is sk-abcdefghijklmnopqrstuvwxyz123456",
                }
            ]
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert "sk-abcdefghijklmnopqrstuvwxyz123456" not in body["content"]
    assert body["raw_metadata"]["secret_redacted"] is True


def test_llm_observability_does_not_record_prompt_text(monkeypatch) -> None:
    monkeypatch.setenv("LLM_ENABLED", "false")
    observability_trace_service.clear()
    marker = "private-prompt-marker-must-not-be-logged"
    response = client.post(
        "/llm/generate",
        json={"messages": [{"role": "user", "content": marker}]},
    )
    assert response.status_code == 200
    traces = observability_trace_service.list_traces()
    request_trace = next(
        trace for trace in traces if trace["name"] == "llm_generate_requested"
    )
    assert request_trace["metadata"]["prompt_recorded"] is False
    assert marker not in str(traces)


def test_existing_brain_think_still_works() -> None:
    response = client.post("/brain/think", json={"raw_input": "open youtube"})

    assert response.status_code == 200
    assert response.json()["message"] == "Opening YouTube now."


def test_existing_secret_scan_still_works() -> None:
    response = client.post(
        "/security/secret-scan",
        json={"text": "sk-abcdefghijklmnopqrstuvwxyz123456"},
    )

    assert response.status_code == 200
    assert response.json()["has_secrets"] is True


def test_existing_retrieval_drivers_still_work() -> None:
    response = client.get("/retrieval/drivers")

    assert response.status_code == 200
    assert len(response.json()) >= 1


def test_llm_providers_endpoint_returns_known_providers() -> None:
    # The Ollama slot resolves through the provider factory, which returns the
    # mock provider when LLM_ENABLED is falsey. Pin it so this test does not
    # depend on the developer shell's environment; patch.dict restores it.
    with patch.dict("os.environ", {"LLM_ENABLED": "true"}):
        response = client.get("/llm/providers")

    assert response.status_code == 200
    providers = {item["provider"] for item in response.json()}
    assert "mock" in providers
    assert "ollama" in providers


def test_llm_models_endpoint_returns_capabilities() -> None:
    response = client.get("/llm/models")

    assert response.status_code == 200
    assert any(item["model"] == "mock-model" for item in response.json())


def test_llm_router_test_returns_safe_decision() -> None:
    response = client.post(
        "/llm/router/test",
        json={"messages": [{"role": "user", "content": "debug this python traceback"}]},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["task_type"] == "coding"
    assert "selected_provider" in body


def test_llm_router_health_and_statistics_work() -> None:
    health = client.get("/llm/router/health")
    statistics = client.get("/llm/router/statistics")

    assert health.status_code == 200
    assert statistics.status_code == 200
    assert "providers" in health.json()
    assert health.json()["omniroute"]["catalog"]["state"] == "disabled"
    assert "total_requests" in statistics.json()


def test_public_router_preview_cannot_authorize_cloud() -> None:
    response = client.post(
        "/llm/router/test",
        json={
            "messages": [
                {
                    "role": "user",
                    "content": "Design a complex software architecture",
                }
            ],
            "metadata": {
                "privacy_class": "cloud_allowed",
                "_cloud_authorized": True,
                "route_id": "attacker-route",
                "base_url": "https://remote.example/v1",
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["selected_provider"] != "omniroute"


def test_omniroute_catalog_is_honestly_disabled() -> None:
    response = client.get("/llm/omniroute/catalog")
    assert response.status_code == 200
    assert response.json()["state"] == "disabled"
