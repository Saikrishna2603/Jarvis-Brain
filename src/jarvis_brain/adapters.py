"""Brain's own capability adapters, extracted from the monolith.

`IntelligenceRouterAdapter`, `DirectLLMProviderAdapter` and
`ToolRegistryAdapter` lived in `app/adapters/builtin.py` alongside Voice,
Vision and Memory adapters, with a shared factory that wired every domain at
once. That factory is exactly the kind of thing the split removes: importing it
pulled Voice, Vision and Memory into whoever needed an LLM adapter.

Each service now owns its own adapters. Class bodies copied verbatim; only
imports changed, plus `ToolRegistryAdapter` taking its registry through a port
because the tool registry is Skills-owned.
"""

from __future__ import annotations

from collections.abc import AsyncIterator
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from jarvis_platform.adapters.base import BaseAdapter
from jarvis_platform.adapters.manager import AdapterManager
from jarvis_platform.adapters.registry import AdapterRegistry
from jarvis_platform.adapters.capabilities import (
    LLM_GENERATE,
    LLM_STREAM,
    LLM_STRUCTURED_OUTPUT,
    TOOL_EXECUTE,
)
from jarvis_platform.adapters.enums import (
    AdapterErrorCategory,
    AdapterExecutionStatus,
    AdapterLifecycleStatus,
    AdapterPrivacy,
    AdapterType,
    AdapterStreamEventType,
)
from jarvis_platform.adapters.errors import AdapterException
from jarvis_platform.adapters.schemas import (
    AdapterExecutionContext,
    AdapterHealth,
    AdapterRequest,
    AdapterMetadata,
    AdapterResult,
    AdapterStreamEvent,
)
from jarvis_platform.cancellation import CancellationToken
from jarvis_platform.streaming import SyncIteratorBridge
from jarvis_platform.schemas.llm import LLMRequest, LLMResponse, LLMStatus
from jarvis_brain.llm.intelligence_router import IntelligenceRouter
from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_brain.llm.provider_registry import LLMProviderRegistry
from jarvis_platform.nervous_system.event_bus import InternalEventBus

#: Skills owns the tool registry; Brain only invokes it.
ToolRegistry = Any


class IntelligenceRouterAdapter(BaseAdapter):
    """Expose the Phase 4 model router as one universal LLM capability adapter."""

    def __init__(self, router: IntelligenceRouter) -> None:
        self.router = router
        super().__init__(
            AdapterMetadata(
                adapter_id="llm-router",
                display_name="LLM Intelligence Router",
                adapter_type=AdapterType.LLM,
                provider="jarvis-intelligence-router",
                capabilities=[LLM_GENERATE, LLM_STRUCTURED_OUTPUT],
                privacy_level=AdapterPrivacy.LOCAL_PREFERRED,
                local_or_cloud="hybrid",
                metadata={
                    "features": ["structured_output", "streaming"],
                    "privacy_enforced_per_request": True,
                },
                description="Phase 4 model/provider policy behind the universal adapter boundary.",
            )
        )

    def execute(self, request: AdapterRequest, context: AdapterExecutionContext) -> AdapterResult:
        if not isinstance(request.payload, LLMRequest):
            raise AdapterException(
                AdapterErrorCategory.INVALID_REQUEST,
                "LLM adapter requires a validated LLMRequest payload.",
            )
        response = self.router.generate(request.payload)
        status = (
            AdapterExecutionStatus.SUCCESS
            if response.status == LLMStatus.SUCCESS
            else AdapterExecutionStatus.UNAVAILABLE
        )
        return AdapterResult(
            adapter_id=self.adapter_id,
            capability=request.capability,
            status=status,
            normalized_output=response,
            provider_metadata={
                "provider": response.provider.value,
                "model": response.model,
                "task_type": response.task_type.value,
            },
            error=None
            if status == AdapterExecutionStatus.SUCCESS
            else self._provider_error(response),
            validation_status="provider_response_validated",
        )

    def health_check(self) -> AdapterHealth:
        health = self.router.list_provider_health()
        available = [item for item in health if item.enabled and item.available]
        return AdapterHealth(
            adapter_id=self.adapter_id,
            status=(
                AdapterLifecycleStatus.READY
                if available
                else AdapterLifecycleStatus.DEGRADED
            ),
            message=(
                f"{len(available)} LLM provider(s) available."
                if available
                else "No configured LLM provider is currently available."
            ),
            details={"available_providers": [item.provider.value for item in available]},
        )

    @staticmethod
    def _provider_error(response: LLMResponse):
        from jarvis_platform.adapters.schemas import AdapterError

        return AdapterError(
            category=AdapterErrorCategory.UNAVAILABLE,
            message=response.error_message or "No LLM provider was available.",
            retryable=False,
        )


class DirectLLMProviderAdapter(BaseAdapter):
    """Compatibility adapter for explicitly injected/test LLM providers."""

    def __init__(self, provider: LLMProvider) -> None:
        self.provider = provider
        super().__init__(
            AdapterMetadata(
                adapter_id=f"llm-provider-{provider.name.value}",
                display_name="Injected LLM Provider",
                adapter_type=AdapterType.LLM,
                provider=provider.name.value,
                capabilities=[LLM_GENERATE, LLM_STRUCTURED_OUTPUT],
                privacy_level=AdapterPrivacy.LOCAL_PREFERRED,
                local_or_cloud="hybrid",
                metadata={"features": ["structured_output", "streaming"]},
                description="Explicitly injected provider behind the universal adapter contract.",
            )
        )

    def execute(self, request: AdapterRequest, context: AdapterExecutionContext) -> AdapterResult:
        if not isinstance(request.payload, LLMRequest):
            raise AdapterException(AdapterErrorCategory.INVALID_REQUEST, "LLM adapter requires LLMRequest.")
        response = self.provider.generate(request.payload)
        return AdapterResult(
            adapter_id=self.adapter_id,
            capability=request.capability,
            status=(AdapterExecutionStatus.SUCCESS if response.status == LLMStatus.SUCCESS else AdapterExecutionStatus.PERMANENT_FAILURE),
            normalized_output=response,
            validation_status="provider_response_validated",
        )

    async def execute_stream(
        self,
        request: AdapterRequest,
        context: AdapterExecutionContext,
    ) -> AsyncIterator[AdapterStreamEvent]:
        if not isinstance(request.payload, LLMRequest):
            raise AdapterException(
                AdapterErrorCategory.INVALID_REQUEST,
                "LLM adapter requires LLMRequest.",
            )
        cancellation = context.metadata.get("_cancellation_token")
        if not isinstance(cancellation, CancellationToken):
            cancellation = CancellationToken()
        content = ""
        provider_metadata: dict[str, Any] = {}
        sequence = 0
        bridge: SyncIteratorBridge[dict] = SyncIteratorBridge(maxsize=32)
        async for item in bridge.iterate(
            lambda: self.provider.generate_stream(request.payload),
            cancellation=cancellation,
        ):
            kind = item.get("type")
            if kind == "token":
                token = str(item.get("content") or "")
                if not token:
                    continue
                content += token
                sequence += 1
                yield AdapterStreamEvent(
                    event_type=AdapterStreamEventType.DELTA,
                    adapter_id=self.adapter_id,
                    capability=request.capability,
                    sequence=sequence,
                    delta={"type": "token", "content": token},
                )
            elif kind == "done":
                provider_metadata = dict(item.get("metadata") or {})
            elif kind == "error":
                raise AdapterException(
                    AdapterErrorCategory.PROVIDER_ERROR,
                    str(item.get("message") or "LLM provider stream failed."),
                )
        response = LLMResponse(
            response_id=str(uuid4()),
            request_id=request.payload.request_id,
            provider=self.provider.name,
            model=self.provider.model,
            task_type=request.payload.task_type,
            status=LLMStatus.SUCCESS,
            content=content,
            raw_metadata=provider_metadata,
        )
        result = AdapterResult(
            adapter_id=self.adapter_id,
            capability=request.capability,
            status=AdapterExecutionStatus.SUCCESS,
            normalized_output=response,
            provider_metadata=provider_metadata,
            validation_status="provider_response_validated",
        )
        sequence += 1
        yield AdapterStreamEvent(
            event_type=AdapterStreamEventType.RESULT,
            adapter_id=self.adapter_id,
            capability=request.capability,
            sequence=sequence,
            result=result,
        )

    def health_check(self) -> AdapterHealth:
        available = self.provider.is_available()
        return AdapterHealth(
            adapter_id=self.adapter_id,
            status=AdapterLifecycleStatus.READY if available else AdapterLifecycleStatus.UNAVAILABLE,
            message="Provider available." if available else "Provider unavailable.",
        )


class ToolRegistryAdapter(BaseAdapter):
    def __init__(self, registry: ToolRegistry) -> None:
        self.tool_registry = registry
        super().__init__(
            AdapterMetadata(
                adapter_id="tool-registry",
                display_name="Tool Registry",
                adapter_type=AdapterType.TOOL,
                provider="jarvis-tools",
                capabilities=[TOOL_EXECUTE],
                permissions=["tool.execute"],
                privacy_level=AdapterPrivacy.LOCAL,
                local_or_cloud="local",
                description="Compatibility adapter; deterministic safety remains authoritative before use.",
            )
        )

    def execute(self, request: AdapterRequest, context: AdapterExecutionContext) -> AdapterResult:
        payload = request.payload if isinstance(request.payload, dict) else {}
        action = str(payload.get("action", "")).strip()
        if not action:
            raise AdapterException(AdapterErrorCategory.INVALID_REQUEST, "Tool execution requires an action.")
        output = self.tool_registry.execute_action(
            action=action,
            target=payload.get("target"),
            payload=payload.get("payload"),
        )
        return AdapterResult(
            adapter_id=self.adapter_id,
            capability=request.capability,
            status=AdapterExecutionStatus.SUCCESS,
            normalized_output=output,
            validation_status="authorized_by_execution_context",
        )




def create_llm_adapter_manager(
    provider_factory: Callable[[str | None], LLMProvider] | None = None,
    intelligence_router: IntelligenceRouter | None = None,
) -> AdapterManager:
    registry = AdapterRegistry()
    router = intelligence_router or IntelligenceRouter(
        provider_registry=LLMProviderRegistry(provider_factory=provider_factory)
    )
    registry.register(IntelligenceRouterAdapter(router))
    manager = AdapterManager(registry=registry, max_retries=0)
    manager.initialize()
    return manager


def create_direct_llm_adapter_manager(provider: LLMProvider) -> AdapterManager:
    registry = AdapterRegistry()
    registry.register(DirectLLMProviderAdapter(provider))
    manager = AdapterManager(registry=registry, max_retries=0)
    manager.initialize()
    return manager


def create_tool_adapter_manager(
    tool_registry: ToolRegistry,
    event_bus: InternalEventBus | None = None,
) -> AdapterManager:
    """`tool_registry` is Skills-owned and arrives through a port."""

    registry = AdapterRegistry()
    registry.register(ToolRegistryAdapter(tool_registry))
    manager = AdapterManager(registry=registry, max_retries=0, event_bus=event_bus)
    manager.initialize()
    return manager
