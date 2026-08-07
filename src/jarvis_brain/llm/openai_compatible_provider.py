import json
import socket
import codecs
from collections.abc import Iterator
from time import perf_counter
import urllib.error
import urllib.request
from uuid import uuid4

from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_platform.schemas.llm import LLMProviderName, LLMRequest, LLMResponse, LLMStatus


class OpenAICompatibleProvider(LLMProvider):
    """Chat provider for any OpenAI-compatible endpoint (e.g. NVIDIA NIM).

    NVIDIA's hosted models at ``https://integrate.api.nvidia.com/v1`` speak the
    OpenAI chat-completions API, so the same adapter serves them, OpenAI, or a
    local vLLM. Requests go to a third-party cloud -- unlike the local Ollama
    provider, conversation text (including transcribed voice) leaves the machine.
    """

    name = LLMProviderName.OPENAI

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str = "https://integrate.api.nvidia.com/v1",
        timeout_seconds: int = 60,
        enabled: bool = False,
        max_response_bytes: int = 4 * 1024 * 1024,
    ) -> None:
        self.model = model
        self.api_key = (api_key or "").strip()
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.enabled = enabled
        self.max_response_bytes = max(1024, max_response_bytes)

    def is_available(self) -> bool:
        """Available when enabled and an API key is configured.

        Presence of a key is checked rather than a live probe so availability
        stays cheap and never adds a billed round trip on every check.
        """
        return bool(self.enabled and self.api_key)

    def generate(self, request: LLMRequest) -> LLMResponse:
        if not self.is_available():
            return self._response(
                request=request,
                status=LLMStatus.DISABLED,
                content="",
                error_message="OpenAI-compatible provider is disabled or missing an API key.",
            )
        started_at = perf_counter()
        try:
            parsed = self._request_json(self._payload(request, stream=False))
            content = (
                parsed.get("choices", [{}])[0]
                .get("message", {})
                .get("content")
            )
            if not isinstance(content, str) or content.strip() == "":
                return self._response(
                    request=request,
                    status=LLMStatus.ERROR,
                    content="",
                    error_message="Provider response did not include message content.",
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
                error_message="Hosted model request timed out.",
            )
        except urllib.error.HTTPError as error:
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message=f"Hosted model returned HTTP {error.code}.",
            )
        except (urllib.error.URLError, OSError):
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message="Could not connect to the hosted model endpoint.",
            )
        except (json.JSONDecodeError, UnicodeDecodeError, ValueError, KeyError, IndexError):
            return self._response(
                request=request,
                status=LLMStatus.ERROR,
                content="",
                error_message="Hosted model returned an unexpected response.",
            )

    def generate_stream(self, request: LLMRequest) -> Iterator[dict]:
        """Yield token deltas from an OpenAI-style Server-Sent Events stream."""
        if not self.is_available():
            yield {"type": "error", "message": "OpenAI-compatible provider is disabled or missing an API key."}
            return
        started_at = perf_counter()
        generated_chars = 0
        try:
            http_request = self._http_request(self._payload(request, stream=True))
            with self._open(http_request) as response:
                total_bytes = 0
                decoder = codecs.getincrementaldecoder("utf-8")()
                pending = ""
                done = False
                cancellation = request.metadata.get("_cancellation_token")
                for raw_chunk in response:
                    raise_if_cancelled = getattr(cancellation, "raise_if_cancelled", None)
                    if callable(raise_if_cancelled):
                        raise_if_cancelled()
                    total_bytes += len(raw_chunk)
                    if total_bytes > self.max_response_bytes:
                        raise ValueError("Provider stream exceeded the configured response limit.")
                    pending += decoder.decode(raw_chunk)
                    lines = pending.split("\n")
                    pending = lines.pop()
                    for line in lines:
                        event = self._parse_sse_line(line, generated_chars)
                        if event is None:
                            continue
                        if event.get("type") == "done_marker":
                            pending = ""
                            done = True
                            break
                        generated_chars += len(str(event.get("content") or ""))
                        yield event
                    if done:
                        break
                pending += decoder.decode(b"", final=True)
                if pending.strip():
                    event = self._parse_sse_line(pending, generated_chars)
                    if event and event.get("type") != "done_marker":
                        generated_chars += len(str(event.get("content") or ""))
                        yield event
            yield {
                "type": "done",
                "metadata": {
                    "openai_compatible": True,
                    "endpoint": self.base_url,
                    "total_latency_ms": round((perf_counter() - started_at) * 1000, 3),
                    "generation_ms": round((perf_counter() - started_at) * 1000, 3),
                    "generated_chars": generated_chars,
                },
            }
        except (TimeoutError, socket.timeout):
            yield {"type": "error", "message": "Hosted model request timed out."}
        except urllib.error.HTTPError as error:
            yield {"type": "error", "message": f"Hosted model returned HTTP {error.code}."}
        except (urllib.error.URLError, OSError, json.JSONDecodeError, UnicodeDecodeError, ValueError):
            yield {"type": "error", "message": "Hosted model stream became unavailable."}

    def warmup(self) -> dict:
        """Validate the key/model with a minimal request; hosted models stay warm."""
        started_at = perf_counter()
        parsed = self._request_json(
            {
                "model": self.model,
                "messages": [{"role": "user", "content": "ok"}],
                "stream": False,
                "max_tokens": 1,
                "temperature": 0.0,
            }
        )
        return {
            "openai_compatible": True,
            "endpoint": self.base_url,
            "total_latency_ms": round((perf_counter() - started_at) * 1000, 3),
            "generated_tokens": int(
                parsed.get("usage", {}).get("completion_tokens") or 0
            ),
        }

    def _payload(self, request: LLMRequest, *, stream: bool) -> dict:
        payload: dict = {
            "model": self.model,
            "messages": [
                {"role": message.role.value, "content": message.content}
                for message in request.messages
            ],
            "stream": stream,
            "temperature": request.temperature,
        }
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        return payload

    def _request_json(self, payload: dict) -> dict:
        with self._open(self._http_request(payload)) as response:
            try:
                raw = response.read(self.max_response_bytes + 1)
            except TypeError:
                raw = response.read()
        if len(raw) > self.max_response_bytes:
            raise ValueError("Provider response exceeded the configured response limit.")
        return json.loads(raw.decode("utf-8"))

    def _open(self, request: urllib.request.Request):
        return urllib.request.urlopen(request, timeout=self.timeout_seconds)

    def _chat_completions_url(self) -> str:
        return f"{self.base_url}/chat/completions"

    def _http_request(self, payload: dict) -> urllib.request.Request:
        return urllib.request.Request(
            self._chat_completions_url(),
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
                "Accept": "application/json",
            },
            method="POST",
        )

    @staticmethod
    def _parse_sse_line(line: str, generated_chars: int) -> dict | None:
        del generated_chars
        normalized = line.strip()
        if not normalized or not normalized.startswith("data:"):
            return None
        data = normalized[len("data:"):].strip()
        if data == "[DONE]":
            return {"type": "done_marker"}
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            return None
        delta = parsed.get("choices", [{}])[0].get("delta", {}).get("content")
        return {"type": "token", "content": delta} if delta else None

    def _performance_metadata(self, parsed: dict, started_at: float) -> dict:
        usage = parsed.get("usage", {}) or {}
        completion_tokens = int(usage.get("completion_tokens") or 0)
        total_ms = (perf_counter() - started_at) * 1000
        tokens_per_second = (
            completion_tokens / (total_ms / 1000) if completion_tokens and total_ms else 0.0
        )
        return {
            "openai_compatible": True,
            "endpoint": self.base_url,
            "total_latency_ms": round(total_ms, 3),
            "generation_ms": round(total_ms, 3),
            "prompt_tokens": int(usage.get("prompt_tokens") or 0),
            "generated_tokens": completion_tokens,
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
