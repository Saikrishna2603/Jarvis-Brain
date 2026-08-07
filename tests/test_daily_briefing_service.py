import time
from datetime import datetime, timedelta

import pytest

from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_brain.briefing.briefing_store import BriefingStore
from jarvis_brain.briefing.collectors import (
    ActivityCollector,
    AgentActivityCollector,
    ApprovalsCollector,
    BriefingCollector,
    MessagesCollector,
    ScheduleCollector,
    SkillsCollector,
    SystemReadinessCollector,
    WorldIntelligenceCollector,
)
from jarvis_brain.briefing.daily_briefing_service import DailyBriefingService
from jarvis_platform.integrations.integration_interface import IntegrationConnector
from jarvis_platform.integrations.integration_registry import IntegrationRegistry
from app.memory.audit_manager import AuditManager
from app.memory.source_registry import SourceRegistry
from app.memory.task_memory_manager import TaskMemoryManager
from app.retrieval.retrieval_registry import RetrievalRegistry
from jarvis_platform.safety.approval_manager import ApprovalManager
from jarvis_platform.schemas.agent_lifecycle import AgentLifecycleStatus, AgentRole
from jarvis_platform.schemas.briefing import (
    BriefingSectionType,
    BriefingSeverity,
    BriefingStatus,
    SkillBriefingItem,
    SkillStatus,
    SourceAvailability,
)
from jarvis_platform.schemas.common import RiskLevel, TaskStatus, utc_now
from app.skills.skill_catalog import SkillCatalog
from app.skills.skill_registry import SkillRegistry


class RealMailConnector(IntegrationConnector):
    """A stand-in for a genuinely configured mail connector."""

    name = "test_mail"
    supported_actions = ["read_email_preview"]
    source_type = "email"
    trust_level = "private"
    real_connector = True

    def preview(self, action, target=None, payload=None):
        """Return safe message summaries, never full bodies."""
        return {
            "items": [
                {
                    "subject": "Deploy failed",
                    "safe_summary": "A deployment failed overnight.",
                    "priority": "urgent",
                    # A real connector may carry a body; the briefing must not
                    # surface it. This field exists precisely to prove that.
                    "body": "SECRET BODY CONTENT that must never be spoken",
                },
                {
                    "subject": "Weekly newsletter",
                    "safe_summary": "A newsletter arrived.",
                    "priority": "informational",
                },
            ]
        }


class RealCalendarConnector(IntegrationConnector):
    """A stand-in for a genuinely configured calendar connector."""

    name = "test_calendar"
    supported_actions = ["read_calendar_preview"]
    source_type = "calendar"
    trust_level = "private"
    real_connector = True

    def preview(self, action, target=None, payload=None):
        """Return calendar events."""
        return {"items": [{"title": "Standup", "summary": "Daily standup at 09:00."}]}


class SlowCollector(BriefingCollector):
    """A collector that never answers in time."""

    section_type = BriefingSectionType.WORLD
    title = "Slow world source"
    source_id = "slow_world"

    def collect(self, period_start, period_end):
        """Block past any sane timeout."""
        time.sleep(5)
        raise AssertionError("This collector should have timed out.")


class ExplodingCollector(BriefingCollector):
    """A collector whose provider is broken."""

    section_type = BriefingSectionType.MESSAGES
    title = "Broken mail provider"
    source_id = "broken_mail"

    def collect(self, period_start, period_end):
        """Fail the way a real provider outage would."""
        raise ConnectionError("mail provider refused the connection")


def healthy_systems():
    """Return an all-healthy subsystem provider."""
    return lambda: [
        {"name": "backend", "label": "Backend", "state": "ok", "detail": "Serving."},
        {"name": "brain", "label": "Brain engine", "state": "ok", "detail": "Active."},
    ]


def build_service(
    collectors=None,
    briefing_store=None,
    collector_timeout_seconds=1.0,
):
    """Build a briefing service over explicit collectors."""
    return DailyBriefingService(
        collectors=collectors if collectors is not None else [],
        briefing_store=briefing_store or BriefingStore(),
        collector_timeout_seconds=collector_timeout_seconds,
    )


def full_collectors(
    integration_registry=None,
    lifecycle_manager=None,
    approval_manager=None,
    task_memory_manager=None,
    skill_registry=None,
    source_registry=None,
):
    """Build the standard collector set with overridable dependencies."""
    integrations = integration_registry or IntegrationRegistry()
    return [
        SystemReadinessCollector(status_provider=healthy_systems()),
        ActivityCollector(
            task_memory_manager=task_memory_manager or TaskMemoryManager(),
            audit_manager=AuditManager(),
        ),
        AgentActivityCollector(lifecycle_manager=lifecycle_manager or AgentLifecycleManager()),
        MessagesCollector(integration_registry=integrations),
        ScheduleCollector(integration_registry=integrations),
        WorldIntelligenceCollector(
            source_registry=source_registry or SourceRegistry(),
            retrieval_registry=RetrievalRegistry(),
        ),
        ApprovalsCollector(approval_manager=approval_manager or ApprovalManager()),
        SkillsCollector(skill_registry=skill_registry or SkillRegistry()),
    ]


# --------------------------------------------------------------------------
# The core promise: no fabricated data.
# --------------------------------------------------------------------------


def test_mock_connectors_are_reported_as_not_configured() -> None:
    """The shipped mock Gmail/calendar connectors must never be read as real.

    MockGmailConnector returns "3 mock unread messages". Reporting that as fact
    is the single worst thing this feature could do, so it is pinned here.
    """
    service = build_service(full_collectors())

    briefing = service.generate()

    messages = briefing.section(BriefingSectionType.MESSAGES)
    schedule = briefing.section(BriefingSectionType.SCHEDULE)

    assert messages.available is False
    assert messages.items == []
    assert messages.unavailable_reason == (
        "No mail connector is configured. Jarvis cannot see your messages."
    )

    assert schedule.available is False
    assert schedule.unavailable_reason == "Calendar access is not configured."

    # The wordings that would be lies:
    spoken = briefing.spoken_summary.lower()
    assert "no calendar events" not in spoken
    assert "no new messages" not in spoken

    # None of the mock connectors' invented content reached the briefing.
    payload = briefing.model_dump_json().lower()
    for invented in ("3 mock unread", "mock gmail", "mock calendar", "security notice"):
        assert invented not in payload

    # And no source that Jarvis reports as available is a mock source.
    for source in briefing.sources:
        assert source.availability == SourceAvailability.AVAILABLE
        assert source.mock_source is False


def test_unconfigured_calendar_says_so_rather_than_reporting_zero_events() -> None:
    """'Not configured' and 'no events' are different claims."""
    service = build_service(full_collectors())

    briefing = service.generate()

    assert "Calendar access is not configured." in briefing.spoken_summary
    unavailable = {source.name for source in briefing.unavailable_sources}
    assert "Calendar and schedule" in unavailable


def test_world_section_is_unavailable_without_a_trusted_source() -> None:
    """No weather/news/finance provider exists, so World must not invent one."""
    service = build_service(full_collectors())

    briefing = service.generate()
    world = briefing.section(BriefingSectionType.WORLD)

    assert world.available is False
    assert "No trusted world intelligence source is configured." in world.summary
    assert world.items == []
    for banned in ("weather", "forecast", "stock", "headline"):
        assert banned not in " ".join(item.title for item in world.items).lower()


def test_world_section_reports_registered_source_without_inventing_content() -> None:
    """A registered source is acknowledged, but its contents are not guessed."""
    sources = SourceRegistry()
    sources.register_source(
        name="Example News",
        source_type="news",
        url="https://example.com/feed",
        trust_level="trusted",
    )
    service = build_service(full_collectors(source_registry=sources))

    world = service.generate().section(BriefingSectionType.WORLD)

    assert world.available is False
    assert "no retrieval driver" in world.unavailable_reason.lower()
    assert world.items == []
    assert world.source_references[0].name == "Example News"


def test_briefing_never_claims_real_data_it_does_not_have() -> None:
    """Every item in the briefing carries a source."""
    service = build_service(full_collectors())

    briefing = service.generate()

    for section in briefing.sections:
        for item in section.items:
            assert item.source, f"{item.title} has no source attribution"


# --------------------------------------------------------------------------
# Partial results: one bad source must not kill the briefing.
# --------------------------------------------------------------------------


def test_partial_briefing_survives_a_timeout_and_a_failure() -> None:
    """A hung world API and a broken mail provider still yield a usable briefing."""
    collectors = [
        SystemReadinessCollector(status_provider=healthy_systems()),
        AgentActivityCollector(lifecycle_manager=AgentLifecycleManager()),
        ApprovalsCollector(approval_manager=ApprovalManager()),
        SlowCollector(),
        ExplodingCollector(),
    ]
    service = build_service(collectors, collector_timeout_seconds=0.2)

    briefing = service.generate()

    assert briefing.partial is True
    # The healthy sections still made it.
    assert briefing.section(BriefingSectionType.SYSTEMS).available is True
    assert briefing.section(BriefingSectionType.AGENTS).available is True
    assert briefing.section(BriefingSectionType.APPROVALS).available is True

    world = briefing.section(BriefingSectionType.WORLD)
    assert world.available is False

    messages = briefing.section(BriefingSectionType.MESSAGES)
    assert messages.available is False
    assert "could not be read" in messages.unavailable_reason

    availabilities = {source.availability for source in briefing.unavailable_sources}
    assert SourceAvailability.TIMED_OUT in availabilities
    assert SourceAvailability.ERROR in availabilities


def test_world_api_failure_does_not_fail_the_briefing() -> None:
    """A world API blowing up is a section-level fact, not a 500."""
    service = build_service(
        [
            SystemReadinessCollector(status_provider=healthy_systems()),
            ExplodingCollector(),
        ],
        collector_timeout_seconds=0.5,
    )

    briefing = service.generate()

    assert briefing.spoken_summary
    assert briefing.section(BriefingSectionType.SYSTEMS).available is True


# --------------------------------------------------------------------------
# The briefing window.
# --------------------------------------------------------------------------


def test_no_previous_briefing_uses_the_configured_recent_window() -> None:
    """With no prior briefing, the window falls back rather than covering history."""
    service = DailyBriefingService(
        collectors=[],
        briefing_store=BriefingStore(),
        recent_window_hours=8,
    )
    now = utc_now()

    briefing = service.generate(now=now)

    assert briefing.period_start == now - timedelta(hours=8)
    assert briefing.metadata["had_previous_briefing"] is False


def test_second_briefing_starts_from_the_last_successful_briefing() -> None:
    """The window runs from the previous briefing once one exists."""
    store = BriefingStore()
    service = build_service([], briefing_store=store)

    first = service.generate()
    second = service.generate()

    assert second.period_start == first.generated_at
    assert second.metadata["had_previous_briefing"] is True


# --------------------------------------------------------------------------
# Agents.
# --------------------------------------------------------------------------


def test_zero_agents_reports_no_activity() -> None:
    """No agents is reported as no agents -- a real, true statement."""
    service = build_service(full_collectors())

    agents = service.generate().section(BriefingSectionType.AGENTS)

    assert agents.available is True
    assert agents.summary == "No agents have run in this period."


def test_multiple_agents_are_narrated_from_lifecycle_records() -> None:
    """Agent narration comes from real records, not inference."""
    manager = AgentLifecycleManager()
    working = manager.create_agent(role=AgentRole.CODER, purpose="implement the API adapter")
    manager.update_status(working.agent_id, AgentLifecycleStatus.WORKING)

    waiting = manager.create_agent(role=AgentRole.SECURITY, purpose="review the firewall change")
    manager.update_status(waiting.agent_id, AgentLifecycleStatus.WAITING_FOR_APPROVAL)

    service = build_service(full_collectors(lifecycle_manager=manager))
    agents = service.generate().section(BriefingSectionType.AGENTS)

    assert agents.severity == BriefingSeverity.WARNING
    summaries = " ".join(item.summary for item in agents.items)
    assert "is waiting for approval" in summaries
    assert waiting.name in agents.summary

    approval_items = [item for item in agents.items if item.approval_required]
    assert len(approval_items) == 1


def test_failed_agent_is_surfaced_with_its_recorded_reason() -> None:
    """A failed agent reports the reason that was recorded, or says none was."""
    manager = AgentLifecycleManager()
    agent = manager.create_agent(role=AgentRole.RESEARCHER, purpose="read the docs")
    manager.fail_agent(agent.agent_id, "the source returned 500")

    service = build_service(full_collectors(lifecycle_manager=manager))
    agents = service.generate().section(BriefingSectionType.AGENTS)

    assert agents.severity == BriefingSeverity.ERROR
    failed = [item for item in agents.items if item.severity == BriefingSeverity.ERROR]
    assert len(failed) == 1
    assert "the source returned 500" in failed[0].summary


def test_completed_agent_within_window_is_reported() -> None:
    """Work finished overnight shows up in the next briefing."""
    manager = AgentLifecycleManager()
    agent = manager.create_agent(role=AgentRole.RESEARCHER, purpose="documentation review")
    manager.complete_agent(agent.agent_id, "documentation review finished")

    service = build_service(full_collectors(lifecycle_manager=manager))
    agents = service.generate().section(BriefingSectionType.AGENTS)

    completed = [item for item in agents.items if item.severity == BriefingSeverity.SUCCESS]
    assert len(completed) == 1
    assert "completed" in completed[0].summary


# --------------------------------------------------------------------------
# Tasks, approvals, messages.
# --------------------------------------------------------------------------


def test_completed_overnight_task_is_reported() -> None:
    """A task completed inside the window is reported from task memory."""
    tasks = TaskMemoryManager()
    task = tasks.create_task(action="run the migration", target="alembic upgrade head")
    tasks.update_task_status(str(task.task_id), TaskStatus.COMPLETED)

    service = build_service(full_collectors(task_memory_manager=tasks))
    activity = service.generate().section(BriefingSectionType.ACTIVITY)

    assert "1 task completed" in activity.summary
    assert activity.items[0].severity == BriefingSeverity.SUCCESS


def test_system_incidents_are_reported_but_routine_audit_chatter_is_not() -> None:
    """Real incidents surface. "intent_resolved" does not."""
    audit = AuditManager()
    audit.record_event(event_type="error", message="The provider test failed.")
    audit.record_event(event_type="intent_resolved", message="Resolved an intent.")
    audit.record_event(event_type="response_generated", message="Answered the user.")

    collectors = [
        ActivityCollector(task_memory_manager=TaskMemoryManager(), audit_manager=audit),
    ]
    section = build_service(collectors).generate().section(BriefingSectionType.ACTIVITY)

    incidents = [item for item in section.items if item.type.value == "incident"]
    assert len(incidents) == 1
    assert incidents[0].summary == "The provider test failed."
    assert incidents[0].severity == BriefingSeverity.ERROR
    assert "1 system incident recorded" in section.summary


def test_waiting_approval_is_reported_and_becomes_a_priority() -> None:
    """Something blocked on the user is surfaced and prioritized."""
    approvals = ApprovalManager()
    approvals.create_approval(
        action="enable_network_service",
        target="world source",
        risk_level=RiskLevel.HIGH,
        reason="This would open outbound network access.",
    )

    service = build_service(full_collectors(approval_manager=approvals))
    briefing = service.generate()
    section = briefing.section(BriefingSectionType.APPROVALS)

    assert briefing.approval_count == 1
    assert "1 approval waiting" in section.summary
    assert section.items[0].approval_required is True
    assert briefing.priority_items[0].title == "enable_network_service"
    assert briefing.priority_items[0].reason


def test_real_mail_connector_is_read_but_bodies_are_never_exposed() -> None:
    """A real connector is read -- and its message bodies still stay private."""
    registry = IntegrationRegistry(connectors=[RealMailConnector()])
    service = build_service(full_collectors(integration_registry=registry))

    briefing = service.generate()
    messages = briefing.section(BriefingSectionType.MESSAGES)

    assert messages.available is True
    assert len(messages.items) == 2
    assert messages.items[0].title == "Deploy failed"
    assert messages.items[0].summary == "A deployment failed overnight."

    # The body must not reach the item, the section, or the spoken line.
    assert "SECRET BODY CONTENT" not in briefing.model_dump_json()
    assert "SECRET BODY CONTENT" not in briefing.spoken_summary


def test_real_calendar_connector_is_read() -> None:
    """A real calendar connector lights the schedule section up automatically."""
    registry = IntegrationRegistry(connectors=[RealCalendarConnector()])
    service = build_service(full_collectors(integration_registry=registry))

    schedule = service.generate().section(BriefingSectionType.SCHEDULE)

    assert schedule.available is True
    assert schedule.items[0].title == "Standup"


# --------------------------------------------------------------------------
# Dedupe, priorities, status.
# --------------------------------------------------------------------------


def test_duplicate_items_are_removed() -> None:
    """The same underlying record must not be reported twice."""
    manager = AgentLifecycleManager()
    agent = manager.create_agent(role=AgentRole.CODER, purpose="build the thing")
    manager.update_status(agent.agent_id, AgentLifecycleStatus.WORKING)

    # Two collectors that both report the same agent record.
    collectors = [
        AgentActivityCollector(lifecycle_manager=manager),
        AgentActivityCollector(lifecycle_manager=manager),
    ]
    service = build_service(collectors)

    briefing = service.generate()
    all_ids = [item.id for section in briefing.sections for item in section.items]

    assert len(all_ids) == len(set(all_ids))


def test_priorities_are_capped_and_carry_reasons() -> None:
    """Jarvis suggests a few priorities, never a wall of them."""
    approvals = ApprovalManager()
    for index in range(6):
        approvals.create_approval(
            action=f"action_{index}",
            target="target",
            risk_level=RiskLevel.HIGH,
            reason="Needs a decision.",
        )

    service = build_service(full_collectors(approval_manager=approvals))
    briefing = service.generate()

    assert len(briefing.priority_items) <= 3
    for priority in briefing.priority_items:
        assert priority.reason


def test_overall_status_reflects_real_state() -> None:
    """A clean system is nominal; a failing one is not."""
    clean = build_service(full_collectors()).generate()
    assert clean.overall_status == BriefingStatus.NOMINAL

    manager = AgentLifecycleManager()
    agent = manager.create_agent(role=AgentRole.CODER, purpose="build")
    manager.fail_agent(agent.agent_id, "compilation failed")
    degraded = build_service(full_collectors(lifecycle_manager=manager)).generate()

    assert degraded.overall_status == BriefingStatus.DEGRADED


def test_greeting_states_date_time_and_mode() -> None:
    """The greeting opens with the real date, time, and Jarvis mode."""
    service = build_service([])
    now = datetime(2026, 7, 12, 7, 30, tzinfo=utc_now().tzinfo)

    briefing = service.generate(now=now, user_name="Sai")

    assert briefing.greeting.startswith("Good morning, Sai.")
    assert "2026" not in briefing.greeting or "July" in briefing.greeting
    assert "standard mode" in briefing.greeting
