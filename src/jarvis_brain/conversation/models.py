from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field

from jarvis_platform.schemas.common import utc_now


class ConversationRole(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"


class ConversationTurn(BaseModel):
    """One safe, in-memory turn retained for follow-up context."""

    role: ConversationRole
    content: str = Field(min_length=1, max_length=8_000)
    created_at: datetime = Field(default_factory=utc_now)


class ConversationResponse(BaseModel):
    """Response returned to an interaction channel such as voice."""

    text: str
    request_id: str
    session_id: str
    turn_id: str | None = None
    mode: str
    latency_trace: dict[str, float] = Field(default_factory=dict)
    runtime_metrics: dict[str, float | int | str | None] = Field(default_factory=dict)
    streaming_sentence_count: int = Field(default=0, ge=0)
    provider: str | None = None
    model: str | None = None
    route: str | None = None
    response_adjacency_action_id: str | None = None
    response_adjacency_expected_type: str | None = None
    response_adjacency_open_reason: str | None = None
    response_adjacency_action_consumed: bool = False
    created_at: datetime = Field(default_factory=utc_now)
