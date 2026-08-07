import urllib.error
import urllib.request
import json

from jarvis_brain.llm.openai_compatible_provider import OpenAICompatibleProvider
from jarvis_brain.llm.omniroute.circuit import GatewayCircuitRegistry
from jarvis_brain.llm.omniroute.config import OmniRouteSettings
from jarvis_brain.llm.omniroute.schemas import OmniRouteRoute
from jarvis_platform.schemas.llm import LLMProviderName
from jarvis_platform.schemas.llm import LLMStatus


class _RejectRedirects(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise urllib.error.HTTPError(req.full_url, code, "Redirect rejected", headers, fp)


class OmniRouteGatewayProvider(OpenAICompatibleProvider):
    """Thin exact-route provider for an operator-run loopback OmniRoute sidecar."""

    name = LLMProviderName.OMNIROUTE

    def __init__(self, route: OmniRouteRoute, settings: OmniRouteSettings) -> None:
        self.route = route
        self.settings = settings
        self._opener = urllib.request.build_opener(_RejectRedirects())
        self.circuit = GatewayCircuitRegistry.for_route(
            route.route_id,
            failure_threshold=settings.circuit_failure_threshold,
            recovery_seconds=settings.circuit_recovery_seconds,
        )
        super().__init__(
            model=route.model_id,
            api_key=settings.api_key,
            base_url=settings.base_url,
            timeout_seconds=settings.timeout_seconds,
            enabled=settings.enabled and route.enabled,
            max_response_bytes=settings.max_response_bytes,
        )

    def is_available(self) -> bool:
        return bool(
            super().is_available()
            and self.route.terms_status == "approved"
            and self.circuit.allow_request()
        )

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/providers/{self.route.provider_id}/chat/completions"

    def generate(self, request):
        response = super().generate(request)
        if response.status == LLMStatus.SUCCESS:
            self.circuit.record_success()
        else:
            self.circuit.record_failure(
                rate_limited="HTTP 429" in (response.error_message or "")
            )
        if (
            response.status == LLMStatus.SUCCESS
            and request.metadata.get("structured_output")
        ):
            if not self.route.supports_structured_output:
                return response.model_copy(
                    update={
                        "status": LLMStatus.ERROR,
                        "content": "",
                        "error_message": "The selected route does not support structured output.",
                    }
                )
            try:
                json.loads(response.content)
            except json.JSONDecodeError:
                return response.model_copy(
                    update={
                        "status": LLMStatus.ERROR,
                        "content": "",
                        "error_message": "The selected route returned invalid structured output.",
                    }
                )
        return response

    def generate_stream(self, request):
        for event in super().generate_stream(request):
            if event.get("type") == "error":
                self.circuit.record_failure(
                    rate_limited="HTTP 429" in str(event.get("message") or "")
                )
            elif event.get("type") == "done":
                self.circuit.record_success()
            if event.get("type") != "done":
                yield event
                continue
            metadata = dict(event.get("metadata") or {})
            metadata.pop("endpoint", None)
            metadata.update(
                {
                    "gateway": "omniroute",
                    "route_id": self.route.route_id,
                    "provider_id": self.route.provider_id,
                    "route_locality": self.route.locality.value,
                    "tools_allowed": False,
                    "internal_fallback_allowed": False,
                    "compression_allowed": False,
                    "circuit_state": self.circuit.snapshot().state.value,
                }
            )
            yield {"type": "done", "metadata": metadata}

    def _payload(self, request, *, stream: bool) -> dict:
        payload = super()._payload(request, stream=stream)
        if request.metadata.get("structured_output") and self.route.supports_structured_output:
            payload["response_format"] = {"type": "json_object"}
        return payload

    def _http_request(self, payload: dict) -> urllib.request.Request:
        request = super()._http_request(payload)
        request.add_header("X-Omniroute-Compression", "off")
        return request

    def _open(self, request):
        return self._opener.open(request, timeout=self.timeout_seconds)

    def _performance_metadata(self, parsed: dict, started_at: float) -> dict:
        metadata = super()._performance_metadata(parsed, started_at)
        metadata.update(
            {
                "gateway": "omniroute",
                "route_id": self.route.route_id,
                "provider_id": self.route.provider_id,
                "route_locality": self.route.locality.value,
                "tools_allowed": False,
                "internal_fallback_allowed": False,
                "compression_allowed": False,
                "circuit_state": self.circuit.snapshot().state.value,
            }
        )
        metadata.pop("endpoint", None)
        return metadata
