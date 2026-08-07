from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from time import perf_counter
from typing import Any

from jarvis_platform.cancellation import CancellationToken


ContextFactory = Callable[[], Awaitable[Any]]


@dataclass(frozen=True, slots=True)
class ContextStage:
    name: str
    factory: ContextFactory
    timeout_seconds: float | None = None


@dataclass(slots=True)
class ContextPipelineResult:
    values: dict[str, Any] = field(default_factory=dict)
    latencies_ms: dict[str, float] = field(default_factory=dict)
    timed_out: list[str] = field(default_factory=list)
    failed: dict[str, str] = field(default_factory=dict)


class ContextPipeline:
    """Run independent context providers concurrently under one fixed budget."""

    def __init__(self, default_budget_seconds: float = 0.35) -> None:
        if default_budget_seconds <= 0:
            raise ValueError("Context latency budget must be positive.")
        self.default_budget_seconds = default_budget_seconds

    async def prepare(
        self,
        stages: Iterable[ContextStage],
        *,
        cancellation: CancellationToken,
        budget_seconds: float | None = None,
    ) -> ContextPipelineResult:
        cancellation.raise_if_cancelled()
        result = ContextPipelineResult()
        stage_list = list(stages)
        if not stage_list:
            return result

        async def run(stage: ContextStage) -> tuple[str, Any, float]:
            started = perf_counter()
            timeout = stage.timeout_seconds or budget
            async with asyncio.timeout(timeout):
                value = await stage.factory()
            return stage.name, value, (perf_counter() - started) * 1000

        budget = budget_seconds or self.default_budget_seconds
        tasks = {
            asyncio.create_task(run(stage), name=f"context-{stage.name}"): stage
            for stage in stage_list
        }
        try:
            done, pending = await asyncio.wait(tasks, timeout=budget)
            for task in done:
                stage = tasks[task]
                try:
                    name, value, latency_ms = task.result()
                    result.values[name] = value
                    result.latencies_ms[name] = latency_ms
                except TimeoutError:
                    result.timed_out.append(stage.name)
                except asyncio.CancelledError:
                    raise
                except Exception as error:
                    result.failed[stage.name] = type(error).__name__
            for task in pending:
                result.timed_out.append(tasks[task].name)
                task.cancel()
            if pending:
                await asyncio.gather(*pending, return_exceptions=True)
            cancellation.raise_if_cancelled()
            return result
        finally:
            for task in tasks:
                if not task.done():
                    task.cancel()
