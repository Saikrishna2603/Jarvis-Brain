from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from datetime import datetime, timedelta

from jarvis_brain.briefing.briefing_store import BriefingStore
from jarvis_brain.briefing.collectors import BriefingCollector
from jarvis_platform.schemas.briefing import (
    BriefingItem,
    BriefingSection,
    BriefingSectionType,
    BriefingSeverity,
    BriefingStatus,
    DailyBriefing,
    PriorityItem,
    SourceAvailability,
    SourceReference,
    UnavailableSource,
)
from jarvis_platform.schemas.common import utc_now
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter


DEFAULT_RECENT_WINDOW_HOURS = 12
DEFAULT_COLLECTOR_TIMEOUT_SECONDS = 5.0
MAX_PRIORITIES = 3


class DailyBriefingService:
    """Assemble the daily briefing from real, configured sources.

    Two properties matter more than completeness:

    * **Partial results are normal.** One provider being slow, broken, or absent
      never fails the briefing. A collector that times out becomes an honest
      unavailable section, and the rest of the briefing is still delivered.
    * **Every claim is attributable.** Items carry a source; sources that could
      not be read are listed in ``unavailable_sources`` rather than dropped, so
      "I don't know" is never silently rendered as "nothing happened".
    """

    def __init__(
        self,
        collectors: list[BriefingCollector],
        briefing_store: BriefingStore | None = None,
        safe_logging_filter: SafeLoggingFilter | None = None,
        recent_window_hours: int = DEFAULT_RECENT_WINDOW_HOURS,
        collector_timeout_seconds: float = DEFAULT_COLLECTOR_TIMEOUT_SECONDS,
        mode_provider=None,
    ) -> None:
        """Create the briefing service over a set of collectors."""
        self.collectors = collectors
        self.briefing_store = briefing_store or BriefingStore()
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()
        self.recent_window_hours = recent_window_hours
        self.collector_timeout_seconds = collector_timeout_seconds
        self.mode_provider = mode_provider

    def generate(self, now: datetime | None = None, user_name: str | None = None) -> DailyBriefing:
        """Generate one briefing, tolerating any individual source failing."""
        generated_at = now or utc_now()
        period_start = self._period_start(generated_at)

        sections = self._collect_sections(period_start, generated_at)
        sections = self._deduplicate(sections)

        unavailable_sources = self._unavailable_sources(sections)
        sources = [
            reference
            for section in sections
            for reference in section.source_references
            if reference.availability == SourceAvailability.AVAILABLE
        ]

        approval_count = self._approval_count(sections)
        warning_count = self._warning_count(sections)
        priorities = self._priorities(sections)
        greeting = self._greeting(generated_at, user_name)
        overall_status = self._overall_status(sections)

        briefing = DailyBriefing(
            generated_at=generated_at,
            period_start=period_start,
            period_end=generated_at,
            greeting=greeting,
            overall_status=overall_status,
            sections=sections,
            priority_items=priorities,
            spoken_summary=self._spoken_summary(
                greeting=greeting,
                generated_at=generated_at,
                sections=sections,
                priorities=priorities,
            ),
            sources=sources,
            unavailable_sources=unavailable_sources,
            approval_count=approval_count,
            warning_count=warning_count,
            generated_from_real_data=True,
            demo=False,
            partial=len(unavailable_sources) > 0,
            metadata={
                "mode": self._mode(),
                "recent_window_hours": self.recent_window_hours,
                "had_previous_briefing": self.briefing_store.last_briefing_at() is not None,
            },
        )

        self.briefing_store.record(briefing)
        return briefing

    def _collect_sections(
        self,
        period_start: datetime,
        period_end: datetime,
    ) -> list[BriefingSection]:
        """Run every collector with a timeout, degrading to honest unavailability."""
        sections: list[BriefingSection] = []
        with ThreadPoolExecutor(max_workers=max(1, len(self.collectors))) as executor:
            futures = {
                executor.submit(collector.collect, period_start, period_end): collector
                for collector in self.collectors
            }
            for future, collector in futures.items():
                try:
                    sections.append(future.result(timeout=self.collector_timeout_seconds))
                except FutureTimeoutError:
                    sections.append(
                        collector.unavailable(
                            f"{collector.title} did not respond in time and was skipped.",
                            availability=SourceAvailability.TIMED_OUT,
                            severity=BriefingSeverity.WARNING,
                        )
                    )
                except Exception as error:  # noqa: BLE001 - one bad source must not kill the briefing
                    sections.append(
                        collector.unavailable(
                            f"{collector.title} could not be read: "
                            f"{self.safe_logging_filter.sanitize_message(str(error))}",
                            availability=SourceAvailability.ERROR,
                            severity=BriefingSeverity.WARNING,
                        )
                    )

        order = list(BriefingSectionType)
        sections.sort(key=lambda section: order.index(section.type))
        return sections

    def _deduplicate(self, sections: list[BriefingSection]) -> list[BriefingSection]:
        """Drop items already reported by an earlier section in this briefing."""
        seen: set[str] = set()
        for section in sections:
            unique: list[BriefingItem] = []
            for item in section.items:
                if item.id in seen:
                    continue
                seen.add(item.id)
                unique.append(item)
            section.items = unique
        return sections

    def _period_start(self, generated_at: datetime) -> datetime:
        """Return the start of the window this briefing covers.

        The window runs from the last successful briefing. With no prior
        briefing, it falls back to a configurable recent window rather than
        claiming to cover all of history.
        """
        last = self.briefing_store.last_briefing_at()
        if last is not None:
            return last
        return generated_at - timedelta(hours=self.recent_window_hours)

    def _greeting(self, generated_at: datetime, user_name: str | None) -> str:
        """Return a time-appropriate greeting stating the date, time, and mode."""
        hour = generated_at.hour
        if hour < 12:
            part = "Good morning"
        elif hour < 18:
            part = "Good afternoon"
        else:
            part = "Good evening"
        if user_name:
            part = f"{part}, {user_name}"
        stamp = generated_at.strftime("%A %d %B, %H:%M UTC")
        return f"{part}. It is {stamp}. Jarvis is in {self._mode()} mode."

    def _mode(self) -> str:
        """Return the current Jarvis mode."""
        if self.mode_provider is None:
            return "standard"
        try:
            return str(self.mode_provider())
        except Exception:  # noqa: BLE001 - the mode is cosmetic, never fail on it
            return "standard"

    def _unavailable_sources(self, sections: list[BriefingSection]) -> list[UnavailableSource]:
        """List every source that could not be read, with its honest reason."""
        unavailable: list[UnavailableSource] = []
        for section in sections:
            for reference in section.source_references:
                if reference.availability == SourceAvailability.AVAILABLE:
                    continue
                unavailable.append(
                    UnavailableSource(
                        source_id=reference.source_id,
                        name=reference.name,
                        section=section.type,
                        availability=reference.availability,
                        reason=reference.detail
                        or section.unavailable_reason
                        or f"{reference.name} is not available.",
                        action_target=reference.url,
                    )
                )
        return unavailable

    def _approval_count(self, sections: list[BriefingSection]) -> int:
        """Count items genuinely waiting on a human decision."""
        return len(
            [
                item
                for section in sections
                for item in section.items
                if item.approval_required
            ]
        )

    def _warning_count(self, sections: list[BriefingSection]) -> int:
        """Count items that warrant attention."""
        return len(
            [
                item
                for section in sections
                for item in section.items
                if item.severity
                in {BriefingSeverity.WARNING, BriefingSeverity.ERROR, BriefingSeverity.CRITICAL}
            ]
        )

    def _overall_status(self, sections: list[BriefingSection]) -> BriefingStatus:
        """Return the overall standing implied by the collected sections."""
        systems = next(
            (section for section in sections if section.type == BriefingSectionType.SYSTEMS),
            None,
        )
        if systems is not None and not systems.available:
            return BriefingStatus.OFFLINE
        if any(section.severity == BriefingSeverity.ERROR for section in sections):
            return BriefingStatus.DEGRADED
        if any(section.severity == BriefingSeverity.WARNING for section in sections):
            return BriefingStatus.ATTENTION
        return BriefingStatus.NOMINAL

    def _priorities(self, sections: list[BriefingSection]) -> list[PriorityItem]:
        """Suggest a small number of priorities, each with its reason.

        These are recommendations, not instructions, and they are capped. A long
        list is not a priority list.
        """
        candidates: list[tuple[int, BriefingItem, BriefingSection]] = []
        for section in sections:
            for item in section.items:
                weight = self._priority_weight(item)
                if weight > 0:
                    candidates.append((weight, item, section))

        candidates.sort(key=lambda entry: entry[0], reverse=True)

        priorities: list[PriorityItem] = []
        for rank, (_, item, section) in enumerate(candidates[:MAX_PRIORITIES], start=1):
            priorities.append(
                PriorityItem(
                    rank=rank,
                    title=item.title,
                    reason=self._priority_reason(item),
                    severity=item.severity,
                    section=section.type,
                    action_label=item.action_label,
                    action_target=item.action_target,
                )
            )
        return priorities

    def _priority_weight(self, item: BriefingItem) -> int:
        """Score how much an item deserves the user's attention first."""
        if item.severity == BriefingSeverity.CRITICAL:
            return 5
        if item.severity == BriefingSeverity.ERROR:
            return 4
        if item.approval_required:
            return 3
        if item.severity == BriefingSeverity.WARNING:
            return 2
        return 0

    def _priority_reason(self, item: BriefingItem) -> str:
        """State plainly why an item is being suggested as a priority."""
        if item.severity in {BriefingSeverity.ERROR, BriefingSeverity.CRITICAL}:
            return f"It failed and nothing else will proceed past it. {item.summary}"
        if item.approval_required:
            return f"It is blocked until you decide. {item.summary}"
        return item.summary

    def _spoken_summary(
        self,
        greeting: str,
        generated_at: datetime,
        sections: list[BriefingSection],
        priorities: list[PriorityItem],
    ) -> str:
        """Build the line Jarvis speaks.

        The spoken summary carries section summaries only -- never message
        bodies. It names what Jarvis cannot see instead of passing an absent
        source off as an empty one, and it runs through the secret filter before
        it can reach a speaker.
        """
        parts: list[str] = [greeting]

        for section in sections:
            if section.type == BriefingSectionType.GREETING:
                continue
            if section.available:
                parts.append(section.summary)

        unavailable = [section for section in sections if not section.available]
        if unavailable:
            reasons = " ".join(
                section.unavailable_reason or f"{section.title} is unavailable."
                for section in unavailable
            )
            parts.append(reasons)

        if priorities:
            ordinals = ["first", "second", "third"]
            for index, priority in enumerate(priorities):
                position = ordinals[index] if index < len(ordinals) else f"number {index + 1}"
                parts.append(f"Your suggested {position} priority is {priority.title}.")

        return self.safe_logging_filter.sanitize_message(" ".join(parts))
