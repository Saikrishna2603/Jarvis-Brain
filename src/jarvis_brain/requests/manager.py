from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from jarvis_brain.requests.context_pipeline import ContextStage
from jarvis_brain.requests.coordinator import (
    AsyncTurnCoordinator,
    TurnExecutionResult,
    TurnProcessor,
)
from jarvis_platform.queues import (
    BoundedAsyncQueue,
    QueueOverflowStrategy,
)
from jarvis_brain.requests.turn import ConversationTurnPriority
from jarvis_platform.nervous_system.event_bus import InternalEventBus


@dataclass(slots=True)
class TurnQueues:
    audio: BoundedAsyncQueue[Any]
    transcript: BoundedAsyncQueue[Any]
    response: BoundedAsyncQueue[Any]
    tts: BoundedAsyncQueue[Any]


class ConversationRuntimeManager:
    """High-level async turn runtime and bounded queue owner."""

    def __init__(
        self,
        *,
        event_bus: InternalEventBus | None = None,
        queue_size: int = 32,
        max_active_turns: int = 32,
    ) -> None:
        self.queue_size = queue_size
        self.coordinator = AsyncTurnCoordinator(
            event_bus=event_bus, max_active_turns=max_active_turns
        )
        self._queues: dict[str, TurnQueues] = {}

    def create_queues(self, turn_id: str) -> TurnQueues:
        queues = TurnQueues(
            audio=BoundedAsyncQueue(
                self.queue_size, overflow=QueueOverflowStrategy.DROP_OLDEST
            ),
            transcript=BoundedAsyncQueue(
                self.queue_size, overflow=QueueOverflowStrategy.BLOCK
            ),
            response=BoundedAsyncQueue(
                self.queue_size, overflow=QueueOverflowStrategy.BLOCK
            ),
            tts=BoundedAsyncQueue(
                self.queue_size, overflow=QueueOverflowStrategy.BLOCK
            ),
        )
        self._queues[turn_id] = queues
        return queues

    async def execute_turn(
        self,
        *,
        user_id: str,
        session_id: str,
        processor: TurnProcessor,
        context_stages: tuple[ContextStage, ...] = (),
        timeout_seconds: float = 30.0,
        context_budget_seconds: float = 0.35,
        priority: ConversationTurnPriority = ConversationTurnPriority.INTERACTIVE,
    ) -> TurnExecutionResult:
        async def wrapped_processor(turn, context, metrics):
            queues = self.create_queues(turn.turn_id)
            try:
                return await processor(turn, context, metrics)
            finally:
                metrics.queue_wait_time_ms = sum(
                    queue.metrics.wait_time_ms
                    for queue in (
                        queues.audio,
                        queues.transcript,
                        queues.response,
                        queues.tts,
                    )
                )
                self._queues.pop(turn.turn_id, None)

        return await self.coordinator.execute(
            user_id=user_id,
            session_id=session_id,
            processor=wrapped_processor,
            context_stages=context_stages,
            timeout_seconds=timeout_seconds,
            context_budget_seconds=context_budget_seconds,
            priority=priority,
        )

    async def cancel_session(
        self, user_id: str, session_id: str, reason: str = "interrupted"
    ) -> bool:
        return await self.coordinator.cancel_session(user_id, session_id, reason)

    def queues_for(self, turn_id: str) -> TurnQueues | None:
        return self._queues.get(turn_id)
