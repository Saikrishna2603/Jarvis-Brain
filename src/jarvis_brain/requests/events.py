from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field

from jarvis_platform.schemas.common import utc_now


class TurnEventType(str, Enum):
    TURN_STARTED = "conversation.turn.started"
    TURN_CANCELLED = "conversation.turn.cancelled"
    CONTEXT_UPDATE = "conversation.context.updated"
    RESPONSE_DELTA = "conversation.response.delta"
    SENTENCE_READY = "conversation.sentence.ready"
    TTS_CHUNK_READY = "conversation.tts.chunk_ready"
    PLAYBACK_STARTED = "conversation.playback.started"
    PLAYBACK_COMPLETED = "conversation.playback.completed"
    TURN_COMPLETED = "conversation.turn.completed"
    TURN_FAILED = "conversation.turn.failed"


class TurnEvent(BaseModel):
    event_id: str = Field(default_factory=lambda: f"turn_event_{uuid4().hex}")
    event_type: TurnEventType
    timestamp: datetime = Field(default_factory=utc_now)
    session_id: str
    user_id: str
    turn_id: str
    sequence: int = Field(ge=1)
    safe_metadata: dict[str, Any] = Field(default_factory=dict)
