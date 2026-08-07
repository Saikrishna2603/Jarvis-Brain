from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, IntEnum
from uuid import uuid4

from jarvis_platform.cancellation import CancellationToken
from jarvis_platform.schemas.common import utc_now


class ConversationTurnState(str, Enum):
    CREATED = "created"
    PREPARING_CONTEXT = "preparing_context"
    RUNNING = "running"
    STREAMING = "streaming"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"
    TIMED_OUT = "timed_out"


class ConversationTurnPriority(IntEnum):
    BACKGROUND = 10
    NORMAL = 50
    INTERACTIVE = 100


@dataclass(slots=True)
class ConversationTurn:
    """One owner-isolated unit of conversational execution."""

    session_id: str
    user_id: str
    timeout_seconds: float
    priority: ConversationTurnPriority = ConversationTurnPriority.INTERACTIVE
    turn_id: str = field(default_factory=lambda: f"turn_{uuid4().hex}")
    created_at: datetime = field(default_factory=utc_now)
    deadline: datetime = field(init=False)
    state: ConversationTurnState = ConversationTurnState.CREATED
    sequence: int = 0
    cancellation: CancellationToken = field(default_factory=CancellationToken)
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.session_id or not self.user_id:
            raise ValueError("Conversation turns require session and user ownership.")
        if self.timeout_seconds <= 0:
            raise ValueError("Conversation turn timeout must be positive.")
        self.deadline = self.created_at + timedelta(seconds=self.timeout_seconds)

    def next_sequence(self) -> int:
        self.sequence += 1
        return self.sequence

    def snapshot(self) -> dict[str, object]:
        return {
            "turn_id": self.turn_id,
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "priority": int(self.priority),
            "created_at": self.created_at.isoformat(),
            "deadline": self.deadline.isoformat(),
            "sequence": self.sequence,
            "cancelled": self.cancellation.cancelled,
            "failure_reason": self.failure_reason,
        }
