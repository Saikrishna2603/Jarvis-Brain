from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable


class TurnCancelledError(asyncio.CancelledError):
    """Raised when an owned conversation turn is cancelled cooperatively."""


class CancellationToken:
    """Cancellation signal readable from async tasks and provider worker threads."""

    def __init__(self) -> None:
        self._cancelled = threading.Event()
        self._reason = "cancelled"
        self._callbacks: list[Callable[[str], None]] = []
        self._lock = threading.RLock()

    @property
    def cancelled(self) -> bool:
        return self._cancelled.is_set()

    @property
    def reason(self) -> str:
        return self._reason

    def cancel(self, reason: str = "cancelled") -> bool:
        with self._lock:
            if self._cancelled.is_set():
                return False
            self._reason = reason.strip() or "cancelled"
            self._cancelled.set()
            callbacks = tuple(self._callbacks)
            self._callbacks.clear()
        for callback in callbacks:
            callback(self._reason)
        return True

    def add_callback(self, callback: Callable[[str], None]) -> None:
        with self._lock:
            if not self._cancelled.is_set():
                self._callbacks.append(callback)
                return
            reason = self._reason
        callback(reason)

    def raise_if_cancelled(self) -> None:
        if self.cancelled:
            raise TurnCancelledError(self.reason)

    async def wait(self, poll_interval_seconds: float = 0.01) -> str:
        while not self.cancelled:
            await asyncio.sleep(poll_interval_seconds)
        return self.reason
