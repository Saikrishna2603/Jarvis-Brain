from __future__ import annotations

import asyncio
from collections import OrderedDict, deque
from collections.abc import Callable
from dataclasses import dataclass, field

from jarvis_brain.conversation.models import (
    ConversationResponse,
    ConversationRole,
    ConversationTurn,
)
from jarvis_brain.engine.brain_engine import BrainEngine
from jarvis_brain.requests.context_pipeline import ContextStage
from jarvis_brain.requests.events import TurnEventType
from jarvis_brain.requests.manager import ConversationRuntimeManager
from jarvis_platform.identity.models import UserIdentity
from jarvis_platform.nervous_system.event_bus import InternalEventBus
from jarvis_platform.security.secret_policy import SecretPolicyEngine
from jarvis_brain.ports import JarvisSpeechStylePolicy
from jarvis_brain.ports import load_jarvis_voice_identity


@dataclass(slots=True)
class _SessionContext:
    turns: deque[ConversationTurn] = field(default_factory=lambda: deque(maxlen=8))
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    brain: object | None = None


class ConversationManager:
    """Own bounded follow-up context while leaving decisions to BrainEngine.

    Context is isolated by user and voice session, kept only in process memory,
    and sanitized before retention. The manager never executes tools or grants
    permissions; it is an interaction layer around the existing Brain boundary.
    """

    def __init__(
        self,
        brain_engine: BrainEngine,
        *,
        max_sessions: int = 64,
        turns_per_session: int = 8,
        secret_policy: SecretPolicyEngine | None = None,
        event_bus: InternalEventBus | None = None,
        runtime: ConversationRuntimeManager | None = None,
        speech_style_policy: JarvisSpeechStylePolicy | None = None,
    ) -> None:
        if max_sessions < 1 or turns_per_session < 2:
            raise ValueError("Conversation limits must be positive.")
        self.brain_engine = brain_engine
        self.max_sessions = max_sessions
        self.turns_per_session = turns_per_session
        self.secret_policy = secret_policy or SecretPolicyEngine()
        self.runtime = runtime or ConversationRuntimeManager(event_bus=event_bus)
        self.speech_style_policy = speech_style_policy or JarvisSpeechStylePolicy(
            load_jarvis_voice_identity()
        )
        self._sessions: OrderedDict[tuple[str, str], _SessionContext] = OrderedDict()

    async def respond(
        self,
        text: str,
        identity: UserIdentity,
        session_id: str,
        *,
        on_response_delta: Callable[[str], None] | None = None,
        conversation_context_id: str | None = None,
        response_adjacency_action_id: str | None = None,
    ) -> ConversationResponse:
        """Process one turn through BrainEngine with owner-scoped context."""
        normalized = text.strip()
        if not normalized:
            raise ValueError("Conversation input must not be empty.")
        context_key = conversation_context_id or session_id
        context = self._context(identity.user_id, context_key)
        async with context.lock:
            async def history_stage():
                return [turn.model_dump(mode="json") for turn in context.turns]

            async def intent_stage():
                resolver = getattr(context.brain, "intent_resolver", None)
                if resolver is None:
                    return {"status": "unavailable"}
                result = await asyncio.to_thread(resolver.resolve, normalized)
                return {
                    "intent_type": result.intent_type,
                    "action": result.action,
                }

            async def process_turn(turn, prepared, metrics):
                history = list(prepared.values.get("conversation_history", []))
                style_context = self.speech_style_policy.context()

                def first_token() -> None:
                    if "first_token" not in metrics.marks:
                        metrics.mark("first_token")

                def safe_delta(sentence: str) -> None:
                    turn.cancellation.raise_if_cancelled()
                    if "first_sentence" not in metrics.marks:
                        metrics.mark("first_sentence")
                    safe_metadata = {"character_count": len(sentence)}
                    self.runtime.coordinator.publish_event(
                        turn, TurnEventType.RESPONSE_DELTA, safe_metadata
                    )
                    self.runtime.coordinator.publish_event(
                        turn, TurnEventType.SENTENCE_READY, safe_metadata
                    )
                    self.runtime.coordinator.publish_event(
                        turn, TurnEventType.TTS_CHUNK_READY, safe_metadata
                    )
                    if on_response_delta is not None:
                        on_response_delta(sentence)

                metadata = {
                    "source": "voice",
                    "user_id": identity.user_id,
                    "conversation_id": session_id,
                    "conversation_history": history,
                    "response_style": "natural_voice",
                    **style_context.safe_metadata(),
                    "_on_response_delta": safe_delta,
                    "_on_first_token": first_token,
                    "_cancellation_token": turn.cancellation,
                    "preclassified_intent": prepared.values.get(
                        "intent_classification", {}
                    ),
                    "response_adjacency_action_id": response_adjacency_action_id,
                }
                response = await asyncio.to_thread(
                    context.brain.process_input,
                    normalized,
                    metadata,
                )
                metrics.mark("generation_completed")
                trace = response.metadata.get("voice_latency_trace", {})
                if isinstance(trace, dict):
                    metrics.adapter_latency_ms = float(
                        trace.get("llm_generation_ms") or 0.0
                    )
                    metrics.token_count = int(trace.get("generated_tokens") or 0)
                return response

            execution = await self.runtime.execute_turn(
                user_id=identity.user_id,
                session_id=session_id,
                processor=process_turn,
                context_stages=(
                    ContextStage("conversation_history", history_stage),
                    ContextStage("intent_classification", intent_stage),
                ),
                timeout_seconds=30.0,
                context_budget_seconds=0.35,
            )
            response = execution.value

            safe_user = self.secret_policy.inspect_text(
                normalized, context="conversation_context"
            ).redacted_text
            context.turns.append(
                ConversationTurn(role=ConversationRole.USER, content=safe_user)
            )
            safe_response = self.secret_policy.inspect_text(
                response.message, context="conversation_context"
            ).redacted_text
            context.turns.append(
                ConversationTurn(role=ConversationRole.ASSISTANT, content=safe_response)
            )
        streaming_sentence_count = int(
            response.metadata.get("streaming_sentence_count") or 0
        )
        if (
            on_response_delta is not None
            and response.message
            and streaming_sentence_count == 0
        ):
            # Deterministic/system responses do not enter the LLM stream. Deliver
            # their already-filtered complete response as one speakable chunk.
            on_response_delta(response.message)
        return ConversationResponse(
            text=response.message,
            request_id=str(response.request_id),
            session_id=session_id,
            turn_id=execution.turn.turn_id,
            mode=response.mode.value,
            latency_trace={
                key: float(value)
                for key, value in dict(
                    response.metadata.get("voice_latency_trace") or {}
                ).items()
                if isinstance(value, (int, float))
            },
            streaming_sentence_count=streaming_sentence_count,
            provider=(
                str(response.metadata["llm_provider"])
                if response.metadata.get("llm_provider")
                else None
            ),
            model=(
                str(response.metadata["llm_model"])
                if response.metadata.get("llm_model")
                else None
            ),
            route=(
                str(response.metadata["selected_route_id"])
                if response.metadata.get("selected_route_id")
                else None
            ),
            response_adjacency_action_id=(
                str(response.metadata["response_adjacency_action_id"])
                if response.metadata.get("response_adjacency_action_id")
                else None
            ),
            response_adjacency_expected_type=(
                str(response.metadata["response_adjacency_expected_type"])
                if response.metadata.get("response_adjacency_expected_type")
                else None
            ),
            response_adjacency_open_reason=(
                str(response.metadata["response_adjacency_open_reason"])
                if response.metadata.get("response_adjacency_open_reason")
                else None
            ),
            response_adjacency_action_consumed=bool(
                response.metadata.get("response_adjacency_action_consumed")
            ),
            runtime_metrics=execution.metrics,
        )

    async def respond_text(
        self,
        text: str,
        identity: UserIdentity,
        session_id: str,
    ) -> str:
        """Voice-pipeline convenience boundary returning only safe display text."""
        response = await self.respond(text, identity, session_id)
        return response.text

    def release(self, user_id: str, session_id: str) -> None:
        """Discard transient conversation context for an ended session."""
        self._sessions.pop((user_id, session_id), None)

    async def cancel_session(
        self, user_id: str, session_id: str, reason: str = "interrupted"
    ) -> bool:
        """Cancel the active owned turn and cooperatively stop provider output."""
        return await self.runtime.cancel_session(user_id, session_id, reason)

    def session_count(self) -> int:
        return len(self._sessions)

    def _context(self, user_id: str, session_id: str) -> _SessionContext:
        key = (user_id, session_id)
        existing = self._sessions.pop(key, None)
        context = existing or _SessionContext(
            turns=deque(maxlen=self.turns_per_session),
            brain=self._conversation_brain(),
        )
        self._sessions[key] = context
        while len(self._sessions) > self.max_sessions:
            self._sessions.popitem(last=False)
        return context

    def _conversation_brain(self):
        fork = getattr(self.brain_engine, "fork_for_conversation", None)
        return fork() if callable(fork) else self.brain_engine
