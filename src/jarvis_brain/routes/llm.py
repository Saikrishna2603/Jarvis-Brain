import os
import json
from collections.abc import Iterator
from typing import Any
from uuid import uuid4

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from jarvis_platform.config import load_app_environment
from jarvis_platform.identity.dependencies import require_user_identity
from jarvis_platform.identity.models import UserIdentity
from jarvis_brain.llm.intelligence_router import intelligence_router
from jarvis_brain.llm.llm_provider_factory import create_llm_provider, create_model_router
from jarvis_brain.llm.model_registry import LLMModelRegistry
from jarvis_brain.llm.omniroute.discovery import OmniRouteDiscoveryClient
from jarvis_brain.llm.safe_llm_service import SafeLLMService
from jarvis_platform.observability.metrics_service import observability_metrics_service
from jarvis_platform.observability.trace_service import observability_trace_service
from jarvis_brain.llm.ollama_provider import OllamaProvider
from jarvis_platform.schemas.llm import LLMMessage, LLMRequest
from jarvis_platform.schemas.llm_topology import RouterTopologySnapshot
from jarvis_platform.security.input_security_gateway import InputSecurityGateway
from jarvis_platform.security.secret_policy import SecretPolicyEngine


router = APIRouter()
input_security_gateway = InputSecurityGateway()
secret_policy = SecretPolicyEngine()


class LLMRouteRequest(BaseModel):
    """Request body for model routing."""

    text: str


class LLMGenerateRequest(BaseModel):
    """Request body for safe LLM generation."""

    messages: list[LLMMessage]
    temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LLMRouterTestRequest(BaseModel):
    """Request body for safe router testing."""

    messages: list[LLMMessage]
    metadata: dict[str, Any] = Field(default_factory=dict)


@router.get("/llm/status")
def llm_status() -> dict[str, Any]:
    """Return configured LLM provider status."""
    load_app_environment()
    router_instance = create_model_router()
    provider = create_llm_provider(model=router_instance.general_model)
    llm_enabled = _env_truthy(os.getenv("LLM_ENABLED"))
    return {
        "provider": provider.name.value,
        "enabled": llm_enabled,
        "general_model": router_instance.general_model,
        "coding_model": router_instance.coding_model,
        "base_url": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "available": provider.is_available(),
        "loaded_config": {
            "LLM_ENABLED": str(os.getenv("LLM_ENABLED", "")).lower(),
            "LLM_PROVIDER": os.getenv("LLM_PROVIDER", ""),
            "VOICE_TTS_PROVIDER": os.getenv("VOICE_TTS_PROVIDER", ""),
            "VOICE_LOCAL_TTS_ENABLED": str(
                os.getenv("VOICE_LOCAL_TTS_ENABLED", "")
            ).lower(),
        },
        "router_enabled": True,
        "omniroute": {
            **intelligence_router.omniroute_settings.safe_status(),
            **intelligence_router.omniroute_registry.safe_summary(),
        },
    }


@router.get("/llm/providers")
def llm_providers() -> list[dict[str, Any]]:
    """Return safe provider health for known providers."""
    return [
        health.model_dump(mode="json")
        for health in intelligence_router.list_provider_health()
    ]


@router.get("/llm/models")
def llm_models() -> list[dict[str, Any]]:
    """Return safe model capability metadata."""
    return [
        model.model_dump(mode="json")
        for model in LLMModelRegistry().list_models()
    ]


@router.get("/llm/router")
def llm_router_status() -> dict[str, Any]:
    """Return router configuration without secrets."""
    config = intelligence_router.config
    omniroute = intelligence_router.omniroute_settings.safe_status()
    return {
        "enabled": config.enabled,
        "default_provider": config.default_provider.value,
        "preferred_providers": [provider.value for provider in config.preferred_providers],
        "retry_count": config.retry_count,
        "timeout_seconds": config.timeout_seconds,
        "latency_budget_ms": config.latency_budget_ms,
        "privacy_default": config.default_privacy,
        "consensus_foundation": True,
        "execution_control": False,
        "omniroute": {
            **omniroute,
            **intelligence_router.omniroute_registry.safe_summary(),
        },
    }


@router.post("/llm/router/test")
def llm_router_test(request: LLMRouterTestRequest) -> dict[str, Any]:
    """Preview a routing decision without generating content."""
    decision = intelligence_router.route_only(
        request.messages, _public_metadata(request.metadata)
    )
    return decision.model_dump(mode="json")


@router.get("/llm/router/health")
def llm_router_health() -> dict[str, Any]:
    """Return router health and provider status."""
    providers = [health.model_dump(mode="json") for health in intelligence_router.list_provider_health()]
    catalog = OmniRouteDiscoveryClient(
        intelligence_router.omniroute_settings,
        intelligence_router.omniroute_registry,
    ).fetch()
    return {
        "status": "ok",
        "providers": providers,
        "available_provider_count": sum(1 for provider in providers if provider.get("available")),
        "omniroute": {
            **intelligence_router.omniroute_settings.safe_status(),
            **intelligence_router.omniroute_registry.safe_summary(),
            "catalog": catalog.model_dump(mode="json"),
        },
    }


@router.get("/llm/omniroute/catalog")
def llm_omniroute_catalog() -> dict[str, Any]:
    """Perform an explicit read-only catalog probe without enabling routes."""
    snapshot = OmniRouteDiscoveryClient(
        intelligence_router.omniroute_settings,
        intelligence_router.omniroute_registry,
    ).fetch()
    return snapshot.model_dump(mode="json")


@router.get("/llm/router/statistics")
def llm_router_statistics() -> dict[str, Any]:
    """Return safe in-memory router telemetry."""
    return intelligence_router.statistics().model_dump(mode="json")


@router.get("/llm/router/topology", response_model=RouterTopologySnapshot)
def llm_router_topology(
    identity: UserIdentity = Depends(require_user_identity),
) -> RouterTopologySnapshot:
    """Return the authoritative safe snapshot for the routing observatory."""
    del identity
    try:
        intelligence_router.refresh_omniroute_registry()
    except (OSError, ValueError):
        # Keep the last validated registry active when an operator edit is invalid.
        pass
    return intelligence_router.routing_observatory.snapshot(intelligence_router)


@router.post("/llm/route")
def route_llm(request: LLMRouteRequest) -> dict[str, str]:
    """Route text to a task type and model."""
    return create_model_router().route(request.text)


@router.post("/llm/generate")
def generate_llm(request: LLMGenerateRequest) -> dict[str, Any]:
    """Generate through SafeLLMService."""
    metadata = _public_metadata(request.metadata)
    metadata["temperature"] = request.temperature
    observability_trace_service.record_trace(
        "llm_generate_requested",
        metadata={
            "message_count": len(request.messages),
            "request_source": str(metadata.get("source") or "api")[:40],
            "structured_output": bool(metadata.get("structured_output")),
            "prompt_recorded": False,
        },
    )
    response = SafeLLMService().generate(
        messages=request.messages,
        metadata=metadata,
    )
    observability_metrics_service.record_model_usage(
        provider=response.provider.value,
        model=response.model,
        task_type=response.task_type.value,
        status=response.status.value,
    )
    total_latency = response.raw_metadata.get("total_latency_ms")
    if isinstance(total_latency, (int, float)):
        observability_metrics_service.record_stage_latency(
            "llm.total", float(total_latency)
        )
    return response.model_dump(mode="json")


@router.post("/llm/generate/stream")
def stream_llm(request: LLMGenerateRequest) -> StreamingResponse:
    """Stream sanitized token deltas through the authoritative router."""
    combined = " ".join(message.content for message in request.messages)
    if input_security_gateway.inspect_input("api_response", combined)["is_suspicious"]:
        raise HTTPException(status_code=400, detail="Unsafe streaming request rejected.")
    sanitized_messages = [
        message.model_copy(
            update={
                "content": secret_policy.inspect_text(
                    message.content, context="llm_request"
                ).redacted_text
            }
        )
        for message in request.messages
    ]
    last_user = next(
        (message.content for message in reversed(sanitized_messages) if message.role.value == "user"),
        sanitized_messages[-1].content,
    )
    model_router = create_model_router()
    task_type = model_router.detect_task_type(last_user, request.metadata)
    model = model_router.select_model(task_type)
    provider = create_llm_provider(model)
    llm_request = LLMRequest(
        request_id=str(uuid4()),
        provider=provider.name,
        model=model,
        task_type=task_type,
        messages=sanitized_messages,
        temperature=request.temperature,
        metadata={
            **_public_metadata(request.metadata),
            "source": str(request.metadata.get("source") or "api_stream")[:40],
            "tools_allowed": False,
        },
    )

    def events() -> Iterator[str]:
        pending = ""
        safety_window = 512
        for item in intelligence_router.generate_stream(llm_request):
            if item.get("type") == "token":
                pending += str(item.get("content", ""))
                if len(pending) <= safety_window:
                    continue
                candidate = pending[:-safety_window]
                inspection = secret_policy.enforce_output_policy(
                    pending, context="llm_stream_response"
                )
                if inspection["blocked"] or inspection["findings"]:
                    pending = inspection["redacted_text"]
                    continue
                yield _sse("token", {"type": "token", "content": candidate})
                pending = pending[-safety_window:]
                continue
            if item.get("type") == "done":
                inspection = secret_policy.enforce_output_policy(
                    pending, context="llm_stream_response"
                )
                safe_content = inspection["redacted_text"]
                if inspection["blocked"]:
                    safe_content = "I cannot display or repeat sensitive credentials."
                if safe_content:
                    yield _sse(
                        "token", {"type": "token", "content": safe_content}
                    )
                pending = ""
                metadata = item.get("metadata", {})
                latency = metadata.get("total_latency_ms")
                if isinstance(latency, (int, float)):
                    observability_metrics_service.record_stage_latency(
                        "llm.stream_total", float(latency)
                    )
            yield _sse(str(item["type"]), item)

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


def _sse(event_type: str, payload: dict) -> str:
    return (
        f"event: llm.{event_type}\n"
        f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"
    )


@router.post("/llm/warmup")
def warmup_llm() -> dict[str, Any]:
    """Explicitly warm the configured local model without generating text."""
    model = create_model_router().general_model
    provider = create_llm_provider(model)
    if not isinstance(provider, OllamaProvider) or not provider.enabled:
        raise HTTPException(status_code=503, detail="Ollama warmup is unavailable.")
    try:
        metadata = provider.warmup()
    except Exception as exc:
        raise HTTPException(status_code=503, detail="Ollama warmup failed.") from exc
    latency = metadata.get("total_latency_ms")
    if isinstance(latency, (int, float)):
        observability_metrics_service.record_stage_latency("llm.warmup", float(latency))
    return {"status": "ready", "provider": provider.name.value, "model": model, "performance": metadata}


def _env_truthy(value: str | None) -> bool:
    """Return True for common truthy environment values."""
    return value is not None and value.strip().lower() in {"1", "true", "yes", "on"}


def _public_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Discard server-owned routing controls from public API input."""
    return {
        key: value
        for key, value in metadata.items()
        if not key.startswith("_")
        and key not in {"cloud_allowed", "provider_credentials", "base_url", "route_id"}
    }
