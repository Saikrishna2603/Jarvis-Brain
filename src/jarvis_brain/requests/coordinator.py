from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from typing import Any

from jarvis_platform.cancellation import TurnCancelledError
from jarvis_brain.requests.context_pipeline import (
    ContextPipeline,
    ContextPipelineResult,
    ContextStage,
)
from jarvis_brain.requests.events import TurnEvent, TurnEventType
from jarvis_brain.requests.metrics import TurnMetricsRecorder
from jarvis_brain.requests.turn import (
    ConversationTurn,
    ConversationTurnPriority,
    ConversationTurnState,
)
from jarvis_platform.nervous_system.event_bus import InternalEventBus
from jarvis_platform.schemas.message_envelope import MessageEnvelope


TurnProcessor = Callable[
    [ConversationTurn, ContextPipelineResult, TurnMetricsRecorder], Awaitable[Any]
]


@dataclass(slots=True)
class TurnExecutionResult:
    turn: ConversationTurn
    value: Any
    context: ContextPipelineResult
    metrics: dict[str, float | int | str | None]


@dataclass(slots=True)
class _ActiveTurn:
    turn: ConversationTurn
    task: asyncio.Task[Any]


class AsyncTurnCoordinator:
    """Own async turn lifecycle without making Brain or authorization decisions."""

    def __init__(
        self,
        *,
        event_bus: InternalEventBus | None = None,
        context_pipeline: ContextPipeline | None = None,
        max_active_turns: int = 32,
        history_limit: int = 100,
    ) -> None:
        if max_active_turns < 1:
            raise ValueError("At least one active turn must be allowed.")
        self.event_bus = event_bus
        self.context_pipeline = context_pipeline or ContextPipeline()
        self._semaphore = asyncio.Semaphore(max_active_turns)
        self._active: dict[str, _ActiveTurn] = {}
        self._session_turns: dict[tuple[str, str], str] = {}
        self._history: deque[dict[str, object]] = deque(maxlen=history_limit)
        self._lock = asyncio.Lock()

    async def execute(
        self,
        *,
        user_id: str,
        session_id: str,
        processor: TurnProcessor,
        context_stages: Iterable[ContextStage] = (),
        timeout_seconds: float = 30.0,
        context_budget_seconds: float = 0.35,
        priority: ConversationTurnPriority = ConversationTurnPriority.INTERACTIVE,
    ) -> TurnExecutionResult:
        turn = ConversationTurn(
            session_id=session_id,
            user_id=user_id,
            timeout_seconds=timeout_seconds,
            priority=priority,
        )
        current_task = asyncio.current_task()
        if current_task is None:
            raise RuntimeError("Conversation turns require an active asyncio task.")
        await self._register(turn, current_task)
        metrics = TurnMetricsRecorder()
        metrics.mark("transcript_ready")
        metrics.mark("turn_started")
        self._emit(turn, TurnEventType.TURN_STARTED)
        context = ContextPipelineResult()
        try:
            async with self._semaphore:
                async with asyncio.timeout(timeout_seconds):
                    turn.cancellation.raise_if_cancelled()
                    turn.state = ConversationTurnState.PREPARING_CONTEXT
                    context = await self.context_pipeline.prepare(
                        context_stages,
                        cancellation=turn.cancellation,
                        budget_seconds=context_budget_seconds,
                    )
                    metrics.mark("context_ready")
                    self._emit(
                        turn,
                        TurnEventType.CONTEXT_UPDATE,
                        {
                            "ready_stages": sorted(context.values),
                            "timed_out_stages": sorted(context.timed_out),
                            "failed_stages": sorted(context.failed),
                        },
                    )
                    turn.cancellation.raise_if_cancelled()
                    turn.state = ConversationTurnState.RUNNING
                    value = await processor(turn, context, metrics)
                    turn.cancellation.raise_if_cancelled()
            turn.state = ConversationTurnState.COMPLETED
            metrics.mark("turn_completed")
            snapshot = metrics.snapshot()
            self._emit(turn, TurnEventType.TURN_COMPLETED, snapshot)
            return TurnExecutionResult(turn, value, context, snapshot)
        except TimeoutError as error:
            turn.cancellation.cancel("deadline_exceeded")
            turn.state = ConversationTurnState.TIMED_OUT
            turn.failure_reason = "deadline_exceeded"
            self._emit(turn, TurnEventType.TURN_FAILED, {"reason": "deadline_exceeded"})
            raise TimeoutError("Conversation turn deadline exceeded.") from error
        except (TurnCancelledError, asyncio.CancelledError) as error:
            turn.cancellation.cancel(turn.cancellation.reason)
            turn.state = ConversationTurnState.CANCELLED
            turn.failure_reason = turn.cancellation.reason
            self._emit(
                turn,
                TurnEventType.TURN_CANCELLED,
                {"reason": turn.cancellation.reason},
            )
            raise TurnCancelledError(turn.cancellation.reason) from error
        except Exception as error:
            turn.state = ConversationTurnState.FAILED
            turn.failure_reason = type(error).__name__
            self._emit(
                turn,
                TurnEventType.TURN_FAILED,
                {"reason": type(error).__name__},
            )
            raise
        finally:
            await self._unregister(turn)

    async def cancel_turn(self, turn_id: str, reason: str = "interrupted") -> bool:
        async with self._lock:
            active = self._active.get(turn_id)
        if active is None:
            return False
        changed = active.turn.cancellation.cancel(reason)
        if active.task is not asyncio.current_task() and not active.task.done():
            active.task.cancel()
        return changed

    async def cancel_session(
        self, user_id: str, session_id: str, reason: str = "interrupted"
    ) -> bool:
        async with self._lock:
            turn_id = self._session_turns.get((user_id, session_id))
        return await self.cancel_turn(turn_id, reason) if turn_id else False

    def active_count(self) -> int:
        return len(self._active)

    def history(self) -> list[dict[str, object]]:
        return list(self._history)

    def publish_event(
        self,
        turn: ConversationTurn,
        event_type: TurnEventType,
        safe_metadata: dict[str, Any] | None = None,
    ) -> None:
        """Publish a safe lifecycle milestone from a trusted turn component."""
        self._emit(turn, event_type, safe_metadata)

    async def _register(
        self, turn: ConversationTurn, task: asyncio.Task[Any]
    ) -> None:
        session_key = (turn.user_id, turn.session_id)
        async with self._lock:
            existing_id = self._session_turns.get(session_key)
        if existing_id:
            await self.cancel_turn(existing_id, "superseded_by_new_turn")
        async with self._lock:
            self._active[turn.turn_id] = _ActiveTurn(turn, task)
            self._session_turns[session_key] = turn.turn_id

    async def _unregister(self, turn: ConversationTurn) -> None:
        async with self._lock:
            self._active.pop(turn.turn_id, None)
            key = (turn.user_id, turn.session_id)
            if self._session_turns.get(key) == turn.turn_id:
                self._session_turns.pop(key, None)
            self._history.append(turn.snapshot())

    def _emit(
        self,
        turn: ConversationTurn,
        event_type: TurnEventType,
        safe_metadata: dict[str, Any] | None = None,
    ) -> None:
        if self.event_bus is None:
            return
        event = TurnEvent(
            event_type=event_type,
            session_id=turn.session_id,
            user_id=turn.user_id,
            turn_id=turn.turn_id,
            sequence=turn.next_sequence(),
            safe_metadata=safe_metadata or {},
        )
        self.event_bus.publish(
            MessageEnvelope(
                sender="async_turn_coordinator",
                recipient="operations",
                content=event.event_type.value,
                metadata={
                    "event_type": event.event_type.value,
                    "turn_event": event.model_dump(mode="json"),
                },
            )
        )
