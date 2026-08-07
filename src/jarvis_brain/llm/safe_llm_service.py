from collections.abc import Callable
from time import perf_counter
from uuid import uuid4

from jarvis_platform.adapters.capabilities import LLM_GENERATE
from jarvis_platform.adapters.enums import AdapterPrivacy
from jarvis_brain.adapters import create_direct_llm_adapter_manager, create_llm_adapter_manager
from jarvis_platform.adapters.manager import AdapterManager
from jarvis_platform.adapters.schemas import AdapterExecutionContext, AdapterRequest
from jarvis_brain.llm.llm_model_router import LLMModelRouter
from jarvis_brain.llm.llm_provider import LLMProvider
from jarvis_brain.llm.llm_provider_factory import create_llm_provider, create_model_router
from jarvis_brain.llm.intelligence_router import IntelligenceRouter
from jarvis_brain.llm.provider_registry import LLMProviderRegistry
from jarvis_platform.schemas.llm import (
    LLMMessage,
    LLMProviderName,
    LLMRequest,
    LLMResponse,
    LLMRole,
    LLMStatus,
)
from jarvis_platform.security.input_security_gateway import InputSecurityGateway
from jarvis_platform.security.secret_policy import SecretPolicyEngine


class SafeLLMService:
    """Sanitized, reasoning-only entry point for local LLM calls."""

    def __init__(
        self,
        model_router: LLMModelRouter | None = None,
        secret_policy: SecretPolicyEngine | None = None,
        input_security_gateway: InputSecurityGateway | None = None,
        provider_factory: Callable[[str | None], LLMProvider] | None = None,
        intelligence_router: IntelligenceRouter | None = None,
        adapter_manager: AdapterManager | None = None,
    ) -> None:
        """Create the service with safe defaults."""
        self.model_router = model_router or create_model_router()
        self.secret_policy = secret_policy or SecretPolicyEngine()
        self.input_security_gateway = input_security_gateway or InputSecurityGateway()
        self._custom_provider_factory = provider_factory is not None
        self.provider_factory = provider_factory or create_llm_provider
        self.intelligence_router = intelligence_router
        self.adapter_manager = adapter_manager

    def generate(
        self,
        messages: list[LLMMessage],
        metadata: dict | None = None,
    ) -> LLMResponse:
        """Generate a sanitized LLM response without granting tool access."""
        metadata = dict(metadata or {})
        last_user_text = self._last_user_text(messages)
        task_type = self.model_router.detect_task_type(last_user_text, metadata)
        selected_model = self.model_router.select_model(task_type)

        injection_report = self.input_security_gateway.inspect_input(
            source="api_response",
            content=" ".join(message.content for message in messages),
        )
        if injection_report["is_suspicious"]:
            return LLMResponse(
                response_id=str(uuid4()),
                request_id=str(uuid4()),
                provider=LLMProviderName.MOCK,
                model=selected_model,
                task_type=task_type,
                status=LLMStatus.SUCCESS,
                content=(
                    "I cannot follow instructions that try to bypass safety "
                    "or reveal secrets."
                ),
                raw_metadata={
                    "prompt_injection_detected": True,
                    "selected_model": selected_model,
                },
            )

        sanitized_messages, secret_redacted = self._sanitize_messages(messages)
        provider = self.provider_factory(selected_model)
        temperature = float(metadata.get("temperature", 0.2))
        request = LLMRequest(
            request_id=str(uuid4()),
            provider=provider.name,
            model=selected_model,
            task_type=task_type,
            messages=sanitized_messages,
            temperature=max(0.0, min(2.0, temperature)),
            metadata={
                **metadata,
                "selected_model": selected_model,
                "secret_redacted": secret_redacted,
                "tools_allowed": False,
            },
        )
        if self.adapter_manager is not None:
            adapter_manager = self.adapter_manager
        elif self._custom_provider_factory and self.intelligence_router is None:
            adapter_manager = create_direct_llm_adapter_manager(provider)
        else:
            adapter_manager = create_llm_adapter_manager(
                provider_factory=self.provider_factory,
                intelligence_router=self.intelligence_router,
            )
        adapter_result = adapter_manager.execute(
            AdapterRequest(
                capability=LLM_GENERATE,
                payload=request,
                required_features=["structured_output"]
                if metadata.get("brain_orchestration") or metadata.get("structured_output")
                else [],
                privacy_requirement=self._adapter_privacy(metadata),
                timeout_seconds=float(metadata.get("timeout_seconds", 30.0)),
                requester="safe_llm_service",
                correlation_id=metadata.get("correlation_id"),
                trace_id=metadata.get("trace_id"),
            ),
            AdapterExecutionContext(
                request_id=request.request_id,
                privacy_policy=self._adapter_privacy(metadata),
                correlation_id=metadata.get("correlation_id"),
                trace_id=metadata.get("trace_id"),
            ),
        )
        if isinstance(adapter_result.normalized_output, LLMResponse):
            response = adapter_result.normalized_output
        else:
            response = LLMResponse(
                response_id=str(uuid4()),
                request_id=request.request_id,
                provider=provider.name,
                model=selected_model,
                task_type=task_type,
                status=LLMStatus.ERROR,
                content="I could not reach an available language model safely.",
                error_message=(
                    adapter_result.error.message
                    if adapter_result.error
                    else "No LLM adapter was available."
                ),
                raw_metadata={"adapter_id": adapter_result.adapter_id},
            )
        output_policy = self.secret_policy.enforce_output_policy(
            response.content,
            context="llm_response",
        )
        content = output_policy["redacted_text"]
        if output_policy["blocked"]:
            content = (
                "I cannot display or repeat sensitive credentials. "
                "I can help you rotate or store them safely."
            )

        raw_metadata = dict(response.raw_metadata)
        raw_metadata.update(
            {
                "selected_model": selected_model,
                "secret_redacted": secret_redacted,
                "output_redacted": bool(output_policy["findings"]),
                "tools_allowed": False,
                "adapter_id": adapter_result.adapter_id,
                "adapter_fallback_used": adapter_result.fallback_used,
                "adapter_fallback_reason": adapter_result.fallback_reason,
            }
        )
        return response.model_copy(
            update={
                "model": selected_model,
                "task_type": task_type,
                "content": content,
                "raw_metadata": raw_metadata,
            }
        )

    def generate_streaming(
        self,
        messages: list[LLMMessage],
        metadata: dict | None = None,
        *,
        on_safe_sentence: Callable[[str], None] | None = None,
    ) -> LLMResponse:
        """Stream provider tokens while releasing only policy-checked sentences.

        This path is intentionally reasoning-only. It never exposes raw provider
        tokens to voice playback: complete sentence buffers pass SecretGuard
        before the callback can observe them. Providers without native streaming
        inherit the base provider's one-chunk compatibility implementation.
        """
        metadata = dict(metadata or {})
        preparation_started = perf_counter()
        last_user_text = self._last_user_text(messages)
        selection_started = perf_counter()
        task_type = self.model_router.detect_task_type(last_user_text, metadata)
        selected_model = self.model_router.select_model(task_type)
        model_selection_ms = (perf_counter() - selection_started) * 1000

        input_safety_started = perf_counter()
        injection_report = self.input_security_gateway.inspect_input(
            source="api_response",
            content=" ".join(message.content for message in messages),
        )
        if injection_report["is_suspicious"]:
            return LLMResponse(
                response_id=str(uuid4()),
                request_id=str(uuid4()),
                provider=LLMProviderName.MOCK,
                model=selected_model,
                task_type=task_type,
                status=LLMStatus.SUCCESS,
                content=(
                    "I cannot follow instructions that try to bypass safety "
                    "or reveal secrets."
                ),
                raw_metadata={
                    "prompt_injection_detected": True,
                    "selected_model": selected_model,
                    "model_selection_ms": round(model_selection_ms, 3),
                    "input_safety_ms": round(
                        (perf_counter() - input_safety_started) * 1000, 3
                    ),
                },
            )

        sanitized_messages, secret_redacted = self._sanitize_messages(messages)
        input_safety_ms = (perf_counter() - input_safety_started) * 1000
        provider = self.provider_factory(selected_model)
        request = LLMRequest(
            request_id=str(uuid4()),
            provider=provider.name,
            model=selected_model,
            task_type=task_type,
            messages=sanitized_messages,
            temperature=max(0.0, min(2.0, float(metadata.get("temperature", 0.2)))),
            max_tokens=_positive_int(metadata.get("max_tokens")),
            metadata={
                **metadata,
                "selected_model": selected_model,
                "secret_redacted": secret_redacted,
                "streaming": True,
                "tools_allowed": False,
            },
        )

        generation_started = perf_counter()
        first_token_ms: float | None = None
        provider_metadata: dict = {}
        raw_content = ""
        pending_sentence = ""
        emitted_sentence_count = 0
        stream_error: str | None = None
        final_provider = provider.name
        final_model = selected_model
        cancellation = metadata.get("_cancellation_token")
        if self._custom_provider_factory and self.intelligence_router is None:
            provider_stream = provider.generate_stream(request)
        else:
            stream_router = self.intelligence_router or IntelligenceRouter(
                provider_registry=LLMProviderRegistry(
                    provider_factory=self.provider_factory
                )
            )
            provider_stream = stream_router.generate_stream(request)
        try:
            for item in provider_stream:
                raise_if_cancelled = getattr(cancellation, "raise_if_cancelled", None)
                if callable(raise_if_cancelled):
                    raise_if_cancelled()
                kind = item.get("type")
                if kind == "token":
                    token = str(item.get("content") or "")
                    if not token:
                        continue
                    if first_token_ms is None:
                        first_token_ms = (perf_counter() - generation_started) * 1000
                        first_token_callback = metadata.get("_on_first_token")
                        if callable(first_token_callback):
                            first_token_callback()
                    raw_content += token
                    pending_sentence += token
                    sentences, pending_sentence = _complete_sentences(pending_sentence)
                    for sentence in sentences:
                        if self._emit_safe_sentence(sentence, on_safe_sentence):
                            emitted_sentence_count += 1
                elif kind == "done":
                    provider_metadata = dict(item.get("metadata") or {})
                    try:
                        final_provider = LLMProviderName(
                            provider_metadata.get("provider", provider.name.value)
                        )
                    except ValueError:
                        final_provider = LLMProviderName.UNKNOWN
                    final_model = str(provider_metadata.get("model") or selected_model)
                elif kind == "error":
                    stream_error = str(item.get("message") or "LLM streaming failed.")
                    break
        finally:
            close = getattr(provider_stream, "close", None)
            if callable(close):
                close()

        if pending_sentence.strip() and self._emit_safe_sentence(
            pending_sentence, on_safe_sentence
        ):
            emitted_sentence_count += 1

        output_safety_started = perf_counter()
        output_policy = self.secret_policy.enforce_output_policy(
            raw_content,
            context="llm_response",
        )
        content = output_policy["redacted_text"]
        if output_policy["blocked"]:
            content = (
                "I cannot display or repeat sensitive credentials. "
                "I can help you rotate or store them safely."
            )
        output_safety_ms = (perf_counter() - output_safety_started) * 1000
        total_generation_ms = (perf_counter() - generation_started) * 1000
        status = (
            LLMStatus.SUCCESS
            if content.strip() and stream_error is None
            else LLMStatus.ERROR
        )
        provider_metadata.update(
            {
                "streaming": True,
                "selected_model": selected_model,
                "secret_redacted": secret_redacted,
                "output_redacted": bool(output_policy["findings"]),
                "tools_allowed": False,
                "model_selection_ms": round(model_selection_ms, 3),
                "input_safety_ms": round(input_safety_ms, 3),
                "output_safety_ms": round(output_safety_ms, 3),
                "llm_first_token_ms": (
                    round(first_token_ms, 3) if first_token_ms is not None else None
                ),
                "llm_stream_total_ms": round(total_generation_ms, 3),
                "safe_sentence_count": emitted_sentence_count,
                "request_preparation_ms": round(
                    (generation_started - preparation_started) * 1000, 3
                ),
            }
        )
        return LLMResponse(
            response_id=str(uuid4()),
            request_id=request.request_id,
            provider=final_provider,
            model=final_model,
            task_type=task_type,
            status=status,
            content=content,
            error_message=stream_error,
            raw_metadata=provider_metadata,
        )

    def _emit_safe_sentence(
        self,
        sentence: str,
        callback: Callable[[str], None] | None,
    ) -> bool:
        normalized = sentence.strip()
        if not normalized:
            return False
        policy = self.secret_policy.enforce_output_policy(
            normalized,
            context="llm_stream_sentence",
        )
        if policy["blocked"]:
            return False
        safe = str(policy["redacted_text"]).strip()
        if not safe:
            return False
        if callback is not None:
            callback(safe)
        return True

    def _adapter_privacy(self, metadata: dict) -> AdapterPrivacy:
        value = str(metadata.get("privacy_class", "local_preferred"))
        aliases = {
            "local_only": AdapterPrivacy.LOCAL,
            "local": AdapterPrivacy.LOCAL,
            "local_preferred": AdapterPrivacy.LOCAL_PREFERRED,
            "cloud_allowed": AdapterPrivacy.CLOUD_ALLOWED,
            "cloud_required": AdapterPrivacy.CLOUD_REQUIRED,
        }
        return aliases.get(value, AdapterPrivacy.LOCAL_PREFERRED)

    def _last_user_text(self, messages: list[LLMMessage]) -> str:
        """Return the most recent user message content."""
        for message in reversed(messages):
            if message.role == LLMRole.USER:
                return message.content
        return messages[-1].content if messages else ""

    def _sanitize_messages(
        self,
        messages: list[LLMMessage],
    ) -> tuple[list[LLMMessage], bool]:
        """Redact secrets from messages before provider calls."""
        sanitized: list[LLMMessage] = []
        secret_redacted = False
        for message in messages:
            scan = self.secret_policy.inspect_text(message.content, context="llm_request")
            if scan.has_secrets:
                secret_redacted = True
            sanitized.append(
                LLMMessage(
                    role=message.role,
                    content=scan.redacted_text,
                )
            )
        return sanitized, secret_redacted


def _complete_sentences(buffer: str) -> tuple[list[str], str]:
    """Return complete speakable sentences and the remaining token buffer."""
    sentences: list[str] = []
    start = 0
    for index, character in enumerate(buffer):
        if character not in ".!?":
            continue
        next_index = index + 1
        if next_index < len(buffer) and buffer[next_index] not in " \n\t\r\"')]}" :
            continue
        sentence = buffer[start:next_index].strip()
        if sentence:
            sentences.append(sentence)
        start = next_index
    return sentences, buffer[start:].lstrip()


def _positive_int(value: object) -> int | None:
    try:
        parsed = int(value) if value is not None else 0
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None
