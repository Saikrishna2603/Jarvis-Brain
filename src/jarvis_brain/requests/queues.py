from __future__ import annotations

import asyncio
from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Generic, TypeVar


T = TypeVar("T")


class QueueOverflowStrategy(str, Enum):
    BLOCK = "block"
    DROP_OLDEST = "drop_oldest"
    DROP_NEWEST = "drop_newest"
    RAISE = "raise"


class QueueOverflowError(RuntimeError):
    pass


@dataclass(slots=True)
class QueueMetrics:
    accepted: int = 0
    dropped: int = 0
    high_watermark: int = 0
    wait_time_ms: float = 0.0


class BoundedAsyncQueue(Generic[T]):
    """Bounded queue with an explicit overflow policy and safe metrics."""

    def __init__(
        self,
        maxsize: int,
        *,
        overflow: QueueOverflowStrategy = QueueOverflowStrategy.BLOCK,
    ) -> None:
        if maxsize < 1:
            raise ValueError("Bounded queues require a positive maximum size.")
        self._queue: asyncio.Queue[T] = asyncio.Queue(maxsize=maxsize)
        self.overflow = overflow
        self.metrics = QueueMetrics()

    @property
    def maxsize(self) -> int:
        return self._queue.maxsize

    def qsize(self) -> int:
        return self._queue.qsize()

    def empty(self) -> bool:
        return self._queue.empty()

    async def put(self, item: T) -> bool:
        started = perf_counter()
        if self._queue.full():
            if self.overflow == QueueOverflowStrategy.DROP_NEWEST:
                self.metrics.dropped += 1
                return False
            if self.overflow == QueueOverflowStrategy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                self.metrics.dropped += 1
            elif self.overflow == QueueOverflowStrategy.RAISE:
                raise QueueOverflowError("Conversation queue capacity was reached.")
        await self._queue.put(item)
        self.metrics.accepted += 1
        self.metrics.wait_time_ms += (perf_counter() - started) * 1000
        self.metrics.high_watermark = max(
            self.metrics.high_watermark, self._queue.qsize()
        )
        return True

    def put_nowait(self, item: T) -> bool:
        if self._queue.full():
            if self.overflow == QueueOverflowStrategy.DROP_NEWEST:
                self.metrics.dropped += 1
                return False
            if self.overflow == QueueOverflowStrategy.DROP_OLDEST:
                try:
                    self._queue.get_nowait()
                    self._queue.task_done()
                except asyncio.QueueEmpty:
                    pass
                self.metrics.dropped += 1
            else:
                raise QueueOverflowError("Conversation queue capacity was reached.")
        self._queue.put_nowait(item)
        self.metrics.accepted += 1
        self.metrics.high_watermark = max(
            self.metrics.high_watermark, self._queue.qsize()
        )
        return True

    async def get(self) -> T:
        return await self._queue.get()

    def get_nowait(self) -> T:
        return self._queue.get_nowait()

    def task_done(self) -> None:
        self._queue.task_done()

    def clear(self) -> int:
        """Remove every pending item and balance unfinished-task accounting."""
        cleared = 0
        while True:
            try:
                self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return cleared
            self._queue.task_done()
            cleared += 1

    async def join(self) -> None:
        await self._queue.join()
