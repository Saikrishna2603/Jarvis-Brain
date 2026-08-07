from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from time import perf_counter

from jarvis_platform.schemas.common import utc_now


@dataclass(slots=True)
class TurnMetricsRecorder:
    """Monotonic per-turn milestones without transcript or prompt content."""

    started: float = field(default_factory=perf_counter)
    marks: dict[str, float] = field(default_factory=dict)
    timestamps: dict[str, datetime] = field(default_factory=dict)
    token_count: int = 0
    queue_wait_time_ms: float = 0.0
    adapter_latency_ms: float = 0.0

    def mark(self, name: str) -> None:
        self.marks[name] = perf_counter()
        self.timestamps[name] = utc_now()

    def add_tokens(self, count: int = 1) -> None:
        self.token_count += max(0, count)

    def elapsed_ms(self, start: str, end: str) -> float | None:
        start_value = self.marks.get(start)
        end_value = self.marks.get(end)
        if start_value is None or end_value is None:
            return None
        return max(0.0, (end_value - start_value) * 1000)

    def snapshot(self) -> dict[str, float | int | str | None]:
        generation_ms = self.elapsed_ms("first_token", "generation_completed")
        token_rate = (
            self.token_count / (generation_ms / 1000)
            if generation_ms and self.token_count
            else None
        )
        milestones = (
            "speech_received",
            "transcript_ready",
            "turn_started",
            "context_ready",
            "first_token",
            "first_sentence",
            "tts_started",
            "audio_started",
            "turn_completed",
        )
        return {
            **{
                f"{name}_time": (
                    self.timestamps[name].isoformat()
                    if name in self.timestamps
                    else None
                )
                for name in milestones
            },
            "speech_to_first_token_ms": self.elapsed_ms(
                "speech_received", "first_token"
            ),
            "speech_to_first_sentence_ms": self.elapsed_ms(
                "speech_received", "first_sentence"
            ),
            "speech_to_first_audio_ms": self.elapsed_ms(
                "speech_received", "audio_started"
            ),
            "transcript_to_turn_start_ms": self.elapsed_ms(
                "transcript_ready", "turn_started"
            ),
            "context_latency_ms": self.elapsed_ms("turn_started", "context_ready"),
            "turn_total_ms": self.elapsed_ms("turn_started", "turn_completed"),
            "token_generation_rate": round(token_rate, 3)
            if token_rate is not None
            else None,
            "queue_wait_time_ms": round(self.queue_wait_time_ms, 3),
            "adapter_latency_ms": round(self.adapter_latency_ms, 3),
        }
