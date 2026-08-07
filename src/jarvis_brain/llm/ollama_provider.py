import json
import socket
from collections.abc import Iterator
from time import perf_counter
import urllib.error
import urllib.request
from uuid import uuid4

from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_platform.schemas.llm import LLMProviderName, LLMRequest, LLMResponse, LLMStatus


class OllamaProvider(LLMProvider):
    """Local Ollama chat provider with safe error handling."""

    name = LLMProviderName.OLLAMA

    def __init__(
        self,
        model: str,
        base_url: str = "http://localhost:11434",
        timeout_seconds: int = 60,
        enabled: bool = False,
        keep_alive: str = "10m",
    ) -> None:
        """Create an Ollama provider.

        Network calls are disabled unless enabled is True.
        """
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.keep_alive = keep_alive

    def is_available(self) -> bool:
        """Return True when Ollama is enabled and responding."""
        if not self.enabled:
            return False

        try:
            with urllib.request.urlopen(
                f"{self.base_url}/api/tags",
                timeout=self.timeout_seconds,
            ) as response:
                return 200 <= int(response.status) < 300
        except Exception:
            return False

    def generate(self, request: LLMRequest) -> LLMResponse:
        """Generate a response through Ollama without raising network errors."""
        if not self.enabled:
            return self._response(
                request=request,
                status=LLMStatus.DISABLED,
                content="",
                error_message="Ollama provider is disabled.",
            )

        payload = self._payload(request, stream=False)

        started_at = perf_counter()
        try:
            parsed = self._request_json(payload)
            content = parsed.get("message", {}).get("content")
            if not isinstance(content, str) or content.strip() == "":
                return self._response(
                    request=request,
                    status=LLMStatus.ERROR,
                    content="",
                    error_message="Ollama response did not include message content.",
                )
            return self._response(
                request=request,
                status=LLMStatus.SUCCESS,
                content=content,
                raw_metadata=self._performance_metadata(parsed, started_at),
            )
        except (TimeoutError, socket.timeout):
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message="Ollama request timed out.",
            )
        except (urllib.error.URLError, OSError):
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message="Could not connect to Ollama.",
            )
        except json.JSONDecodeError:
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message="Ollama returned invalid JSON.",
            )

    def generate_stream(self, request: LLMRequest) -> Iterator[dict]:
        """Yield safe token deltas and one final performance record."""
        if not self.enabled:
            yield {"type": "error", "message": "Ollama provider is disabled."}
            return
        started_at = perf_counter()
        http_request = self._http_request(self._payload(request, stream=True))
        try:
            with urllib.request.urlopen(http_request, timeout=self.timeout_seconds) as response:
                for raw_line in response:
                    if not raw_line.strip():
                        continue
                    parsed = json.loads(raw_line.decode("utf-8"))
                    content = parsed.get("message", {}).get("content", "")
                    if content:
                        yield {"type": "token", "content": content}
                    if parsed.get("done"):
                        yield {
                            "type": "done",
                            "metadata": self._performance_metadata(parsed, started_at),
                        }
        except (TimeoutError, socket.timeout):
            yield {"type": "error", "message": "Ollama request timed out."}
        except (urllib.error.URLError, OSError, json.JSONDecodeError):
            yield {"type": "error", "message": "Ollama stream became unavailable."}

    def warmup(self) -> dict:
        """Load the configured model using an empty generation."""
        started_at = perf_counter()
        parsed = self._request_json(
            {
                "model": self.model,
                "messages": [],
                "stream": False,
                "keep_alive": self.keep_alive,
            }
        )
        return self._performance_metadata(parsed, started_at)

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict:
        options = {"temperature": request.temperature}
        if request.max_tokens is not None:
            options["num_predict"] = request.max_tokens
        return {
            "model": self.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "stream": stream,
            "keep_alive": self.keep_alive,
            "options": options,
        }

    def _request_json(self, payload: dict) -> dict:
        with urllib.request.urlopen(
            self._http_request(payload), timeout=self.timeout_seconds
        ) as response:
            return json.loads(response.read().decode("utf-8"))

    def _http_request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

    def _performance_metadata(self, parsed: dict, started_at: float) -> dict:
        eval_count = int(parsed.get("eval_count") or 0)
        eval_duration_ns = int(parsed.get("eval_duration") or 0)
        tokens_per_second = (
            eval_count / (eval_duration_ns / 1_000_000_000)
            if eval_count and eval_duration_ns
            else 0.0
        )
        return {
            "ollama": True,
            "keep_alive": self.keep_alive,
            "total_latency_ms": round((perf_counter() - started_at) * 1000, 3),
            "model_load_ms": round(int(parsed.get("load_duration") or 0) / 1_000_000, 3),
            "prompt_processing_ms": round(int(parsed.get("prompt_eval_duration") or 0) / 1_000_000, 3),
            "generation_ms": round(eval_duration_ns / 1_000_000, 3),
            "prompt_tokens": int(parsed.get("prompt_eval_count") or 0),
            "generated_tokens": eval_count,
            "tokens_per_second": round(tokens_per_second, 3),
        }

    def _response(
        self,
        request: LLMRequest,
        status: LLMStatus,
        content: str,
        error_message: str | None = None,
        raw_metadata: dict | None = None,
    ) -> LLMResponse:
        """Build a standard LLMResponse."""
        return LLMResponse(
            response_id=str(uuid4()),
            request_id=request.request_id,
            provider=self.name,
            model=self.model,
            task_type=request.task_type,
            status=status,
            content=content,
            error_message=error_message,
            raw_metadata=raw_metadata or {},
        )
