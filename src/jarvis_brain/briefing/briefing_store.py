from datetime import datetime

from jarvis_platform.schemas.briefing import BriefingRecord, DailyBriefing


class BriefingStore:
    """Remember that a briefing happened, not what was in it.

    This store answers exactly one question for the next briefing: "since when?"
    So it keeps identifiers, timestamps, availability, and counts.

    It deliberately holds no message bodies, no calendar contents, no audio, no
    secrets, and no model reasoning. If a field could leak private content, it
    does not belong here -- the detail stays in the system that owns it, and the
    briefing links out to it instead.
    """

    def __init__(self, max_records: int = 30) -> None:
        """Create an in-memory store of recent briefing metadata."""
        self.max_records = max(1, max_records)
        self._records: list[BriefingRecord] = []

    def record(self, briefing: DailyBriefing) -> BriefingRecord:
        """Store safe metadata for a delivered briefing."""
        record = BriefingRecord(
            briefing_id=briefing.briefing_id,
            generated_at=briefing.generated_at,
            period_start=briefing.period_start,
            period_end=briefing.period_end,
            overall_status=briefing.overall_status,
            section_availability={
                section.type.value: section.available for section in briefing.sections
            },
            item_ids=[item.id for section in briefing.sections for item in section.items],
            approval_ids=[
                item.id
                for section in briefing.sections
                for item in section.items
                if item.approval_required
            ],
            source_ids=[reference.source_id for reference in briefing.sources],
            unavailable_source_ids=[
                source.source_id for source in briefing.unavailable_sources
            ],
            approval_count=briefing.approval_count,
            warning_count=briefing.warning_count,
        )
        self._records.append(record)
        if len(self._records) > self.max_records:
            self._records = self._records[-self.max_records :]
        return record

    def last_briefing_at(self) -> datetime | None:
        """Return when the last briefing was generated, or None if never."""
        if not self._records:
            return None
        return self._records[-1].generated_at

    def last_record(self) -> BriefingRecord | None:
        """Return the most recent briefing record."""
        return self._records[-1] if self._records else None

    def get(self, briefing_id: str) -> BriefingRecord | None:
        """Return one briefing record by id."""
        for record in reversed(self._records):
            if record.briefing_id == briefing_id:
                return record
        return None

    def history(self) -> list[BriefingRecord]:
        """Return recent briefing records, oldest first."""
        return list(self._records)

    def mark_replayed(self, briefing_id: str) -> BriefingRecord | None:
        """Record that the user replayed a briefing."""
        record = self.get(briefing_id)
        if record is None:
            return None
        record.replayed_count += 1
        return record

    def mark_spoken(self, briefing_id: str) -> BriefingRecord | None:
        """Record that a briefing was spoken aloud."""
        record = self.get(briefing_id)
        if record is None:
            return None
        record.spoken = True
        return record

    def mark_dismissed(self, briefing_id: str) -> BriefingRecord | None:
        """Record that the user dismissed a briefing."""
        record = self.get(briefing_id)
        if record is None:
            return None
        record.dismissed = True
        return record

    def clear(self) -> None:
        """Remove all briefing records."""
        self._records.clear()
