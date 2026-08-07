from __future__ import annotations

import asyncio
from collections.abc import Callable, Iterator
from concurrent.futures import TimeoutError as FutureTimeoutError
from typing import Generic, TypeVar

from jarvis_platform.cancellation import CancellationToken
from jarvis_platform.queues import BoundedAsyncQueue


T = TypeVar("T")
_END = object()


class SyncIteratorBridge(Generic[T]):
    """Adapt a blocking provider iterator to a bounded async consumer."""

    def __init__(self, maxsize: int = 32) -> None:
        self.maxsize = maxsize

    async def iterate(
        self,
        iterator_factory: Callable[[], Iterator[T]],
        *,
        cancellation: CancellationToken,
    ):
        queue: BoundedAsyncQueue[object] = BoundedAsyncQueue(self.maxsize)
        loop = asyncio.get_running_loop()

        def push(item: object) -> bool:
            future = asyncio.run_coroutine_threadsafe(queue.put(item), loop)
            while not cancellation.cancelled:
                try:
                    future.result(timeout=0.05)
                    return True
                except FutureTimeoutError:
                    continue
            future.cancel()
            return False

        def produce() -> None:
            iterator = iterator_factory()
            try:
                for item in iterator:
                    if cancellation.cancelled:
                        break
                    if not push(item):
                        break
            except Exception as error:
                push(error)
            finally:
                close = getattr(iterator, "close", None)
                if callable(close):
                    close()
                if not cancellation.cancelled:
                    push(_END)

        producer = asyncio.create_task(asyncio.to_thread(produce))
        completed = False
        try:
            while True:
                cancellation.raise_if_cancelled()
                item = await queue.get()
                queue.task_done()
                if item is _END:
                    completed = True
                    return
                if isinstance(item, Exception):
                    raise item
                yield item
        finally:
            if not completed:
                cancellation.cancel("stream_consumer_closed")
            await asyncio.gather(producer, return_exceptions=True)
