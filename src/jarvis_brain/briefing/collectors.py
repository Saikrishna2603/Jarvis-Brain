from abc import ABC, abstractmethod
from datetime import datetime

from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_platform.integrations.integration_registry import IntegrationRegistry
from jarvis_brain.ports import AuditManager
from jarvis_brain.ports import SourceRegistry
from jarvis_brain.ports import TaskMemoryManager
from jarvis_brain.ports import RetrievalRegistry
from jarvis_platform.safety.approval_manager import ApprovalManager
from jarvis_platform.schemas.agent_lifecycle import AgentLifecycleStatus, AgentRecord
from jarvis_platform.schemas.briefing import (
    BriefingItem,
    BriefingItemType,
    BriefingSection,
    BriefingSectionType,
    BriefingSeverity,
    EvidenceStanding,
    SkillStatus,
    SourceAvailability,
    SourceReference,
)
from jarvis_platform.schemas.common import ApprovalStatus, RiskLevel, TaskStatus, utc_now
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter
from jarvis_brain.ports import SkillRegistry


class BriefingCollector(ABC):
    """One source of briefing content.

    A collector has exactly two honest outcomes: it reports what it really saw,
    or it reports that it could not see. It must never return an empty result to
    stand in for an absent source -- "no events" and "no calendar" are different
    claims, and only one of them is true here.
    """

    section_type: BriefingSectionType
    title: str
    source_id: str

    @abstractmethod
    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect one section for the window between period_start and period_end."""
        raise NotImplementedError

    def unavailable(
        self,
        reason: str,
        availability: SourceAvailability = SourceAvailability.NOT_CONFIGURED,
        severity: BriefingSeverity = BriefingSeverity.INFO,
        action_target: str | None = None,
    ) -> BriefingSection:
        """Return an honest 'I cannot see this' section."""
        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=reason,
            available=False,
            unavailable_reason=reason,
            severity=severity,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name=self.title,
                    source_type=self.section_type.value,
                    availability=availability,
                    detail=reason,
                    url=action_target,
                )
            ],
        )


class SystemReadinessCollector(BriefingCollector):
    """Report which Jarvis subsystems are actually ready.

    Healthy systems are summarized in one line. Only degraded, unavailable, or
    unconfigured services are itemized -- a wall of green cards is noise, and it
    trains the user to stop reading.
    """

    section_type = BriefingSectionType.SYSTEMS
    title = "System readiness"
    source_id = "system_readiness"

    def __init__(self, status_provider) -> None:
        """Create the collector over a callable returning subsystem states."""
        self.status_provider = status_provider

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect real subsystem readiness."""
        subsystems = self.status_provider()
        degraded = [item for item in subsystems if item["state"] in {"degraded", "error"}]
        unconfigured = [item for item in subsystems if item["state"] == "not_configured"]
        healthy = [item for item in subsystems if item["state"] == "ok"]

        items: list[BriefingItem] = []
        for item in [*degraded, *unconfigured]:
            items.append(
                BriefingItem(
                    id=f"system_{item['name']}",
                    title=item["label"],
                    summary=item["detail"],
                    type=BriefingItemType.SYSTEM,
                    # A service that was never configured is a standing fact, not
                    # something that went wrong overnight. Rating it a warning
                    # would push "Jarvis has no microphone" into the priority
                    # list every single morning and drown out real failures.
                    severity=self._severity(item["state"]),
                    source=self.source_id,
                    occurred_at=period_end,
                    action_label="Open diagnostics",
                    action_target="/settings?tab=diagnostics",
                    safe_metadata={"state": item["state"]},
                )
            )

        if not degraded and not unconfigured:
            summary = "All required local services are operational."
            severity = BriefingSeverity.SUCCESS
        elif degraded:
            summary = (
                f"{len(healthy)} services operational. "
                f"{len(degraded)} {'needs' if len(degraded) == 1 else 'need'} attention: "
                f"{', '.join(item['label'] for item in degraded)}."
            )
            severity = BriefingSeverity.ERROR
        else:
            summary = (
                "All required local services are operational. "
                f"{len(unconfigured)} optional service"
                f"{' is' if len(unconfigured) == 1 else 's are'} not configured."
            )
            severity = BriefingSeverity.INFO

        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=summary,
            items=items,
            severity=severity,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name="Jarvis subsystem health",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                )
            ],
        )

    def _severity(self, state: str) -> BriefingSeverity:
        """Map a subsystem state onto briefing severity."""
        if state == "error":
            return BriefingSeverity.ERROR
        if state == "degraded":
            return BriefingSeverity.WARNING
        return BriefingSeverity.INFO


class ActivityCollector(BriefingCollector):
    """Report task and system activity within the briefing window."""

    section_type = BriefingSectionType.ACTIVITY
    title = "Recent activity"
    source_id = "task_memory"

    TERMINAL_STATUSES = {
        TaskStatus.COMPLETED: BriefingSeverity.SUCCESS,
        TaskStatus.CANCELLED: BriefingSeverity.INFO,
    }

    # Audit events worth waking the user up for. Everything else the audit log
    # records ("intent_resolved", "response_generated") is routine chatter and
    # would bury the incidents that actually matter.
    INCIDENT_EVENTS = {
        "error": BriefingSeverity.ERROR,
        "output_blocked_by_secret_policy": BriefingSeverity.WARNING,
        "unsafe_cybersecurity_request_blocked": BriefingSeverity.WARNING,
        "secret_detected": BriefingSeverity.WARNING,
        "world_intelligence_error": BriefingSeverity.WARNING,
        "context_reference_failed": BriefingSeverity.WARNING,
    }

    def __init__(
        self,
        task_memory_manager: TaskMemoryManager,
        audit_manager: AuditManager,
        safe_logging_filter: SafeLoggingFilter | None = None,
    ) -> None:
        """Create the collector over real task memory and the audit log."""
        self.task_memory_manager = task_memory_manager
        self.audit_manager = audit_manager
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect tasks that changed inside the window."""
        tasks = [
            task
            for task in self.task_memory_manager.get_all_tasks()
            if period_start <= task.updated_at <= period_end
        ]

        completed = [task for task in tasks if task.status == TaskStatus.COMPLETED]
        cancelled = [task for task in tasks if task.status == TaskStatus.CANCELLED]
        pending = [
            task
            for task in self.task_memory_manager.get_all_tasks()
            if task.status in {TaskStatus.PENDING, TaskStatus.IN_PROGRESS}
        ]

        items: list[BriefingItem] = []
        for task in [*completed, *cancelled]:
            items.append(
                BriefingItem(
                    id=f"task_{task.task_id}",
                    title=self.safe_logging_filter.sanitize_message(task.title),
                    summary=self._task_summary(task),
                    type=BriefingItemType.TASK,
                    severity=self.TERMINAL_STATUSES.get(task.status, BriefingSeverity.INFO),
                    source=self.source_id,
                    occurred_at=task.updated_at,
                    action_label="Open activity",
                    action_target="/activity",
                    safe_metadata={"status": task.status.value},
                )
            )

        incidents = self._incidents(period_start, period_end)
        items.extend(incidents)

        summary = self._summary(completed, cancelled, pending, incidents)
        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=summary,
            items=items,
            severity=BriefingSeverity.ERROR if incidents else BriefingSeverity.INFO,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name="Task memory",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                ),
                SourceReference(
                    source_id="audit_log",
                    name="Audit log",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                ),
            ],
        )

    def _incidents(self, period_start: datetime, period_end: datetime) -> list[BriefingItem]:
        """Return meaningful system incidents recorded inside the window."""
        incidents: list[BriefingItem] = []
        for event in self.audit_manager.get_all_events():
            severity = self.INCIDENT_EVENTS.get(event.event_name)
            if severity is None:
                continue
            if not period_start <= event.created_at <= period_end:
                continue
            message = str(event.details.get("message", event.event_name))
            incidents.append(
                BriefingItem(
                    id=f"incident_{event.event_id}",
                    title=event.event_name.replace("_", " ").capitalize(),
                    # The audit manager already sanitized this on the way in; it
                    # is filtered again here because it is about to be spoken.
                    summary=self.safe_logging_filter.sanitize_message(message),
                    type=BriefingItemType.INCIDENT,
                    severity=severity,
                    source="audit_log",
                    occurred_at=event.created_at,
                    action_label="Open activity",
                    action_target="/activity?tab=audit",
                    safe_metadata={"event": event.event_name},
                )
            )
        return incidents

    def _task_summary(self, task) -> str:
        """Return a safe one-line summary for a task."""
        description = task.description or "No further detail was recorded."
        return self.safe_logging_filter.sanitize_message(
            f"{task.status.value.replace('_', ' ').capitalize()}. {description}"
        )

    def _summary(
        self,
        completed: list,
        cancelled: list,
        pending: list,
        incidents: list,
    ) -> str:
        """Return an honest activity summary, including the genuinely-empty case."""
        parts: list[str] = []
        if completed:
            parts.append(f"{len(completed)} task{'' if len(completed) == 1 else 's'} completed")
        if cancelled:
            parts.append(f"{len(cancelled)} cancelled")
        if pending:
            parts.append(f"{len(pending)} still pending")
        if incidents:
            parts.append(
                f"{len(incidents)} system incident{'' if len(incidents) == 1 else 's'} recorded"
            )
        if not parts:
            return "No task activity was recorded in this period."
        return f"{', '.join(parts)}."


class AgentActivityCollector(BriefingCollector):
    """Report what agents are really doing, from lifecycle records only.

    Every line here is backed by an ``AgentRecord``. Jarvis narrates status,
    role, and assigned purpose. It does not infer hidden work or speculate about
    what an agent is "probably" doing.
    """

    section_type = BriefingSectionType.AGENTS
    title = "Agent activity"
    source_id = "agent_lifecycle"

    WAITING_STATUSES = {
        AgentLifecycleStatus.WAITING_FOR_INPUT,
        AgentLifecycleStatus.WAITING_FOR_APPROVAL,
    }

    def __init__(self, lifecycle_manager: AgentLifecycleManager) -> None:
        """Create the collector over the agent lifecycle manager."""
        self.lifecycle_manager = lifecycle_manager

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect agent lifecycle state for the window."""
        active = list(self.lifecycle_manager.active_agents.values())
        completed = [
            agent
            for agent in self.lifecycle_manager.completed_agents.values()
            if self._ended_within(agent, period_start, period_end)
        ]
        failed = [
            agent
            for agent in self.lifecycle_manager.failed_agents.values()
            if self._ended_within(agent, period_start, period_end)
        ]

        items = [
            *(self._item(agent) for agent in active),
            *(self._item(agent) for agent in failed),
            *(self._item(agent) for agent in completed),
        ]

        waiting = [agent for agent in active if agent.status in self.WAITING_STATUSES]
        severity = BriefingSeverity.INFO
        if failed:
            severity = BriefingSeverity.ERROR
        elif waiting:
            severity = BriefingSeverity.WARNING

        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=self._summary(active, completed, failed, waiting),
            items=items,
            severity=severity,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name="Agent lifecycle records",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                )
            ],
        )

    def _ended_within(
        self,
        agent: AgentRecord,
        period_start: datetime,
        period_end: datetime,
    ) -> bool:
        """Return True when an agent finished inside the briefing window."""
        ended_at = agent.ended_at or agent.updated_at
        return period_start <= ended_at <= period_end

    def _item(self, agent: AgentRecord) -> BriefingItem:
        """Build one narration item from a real agent record."""
        return BriefingItem(
            id=f"agent_{agent.agent_id}",
            title=agent.name,
            summary=self._narrate(agent),
            type=BriefingItemType.AGENT,
            severity=self._severity(agent),
            source=self.source_id,
            occurred_at=agent.ended_at or agent.updated_at,
            action_label="Open agent",
            action_target=f"/agentsworkspace?agent={agent.agent_id}",
            approval_required=agent.status == AgentLifecycleStatus.WAITING_FOR_APPROVAL,
            safe_metadata={
                "status": agent.status.value,
                "role": agent.role.value,
                "progress_percent": agent.progress_percent,
            },
        )

    def _narrate(self, agent: AgentRecord) -> str:
        """Narrate an agent from its recorded state alone."""
        if agent.status == AgentLifecycleStatus.WAITING_FOR_APPROVAL:
            return f"{agent.name} is waiting for approval."
        if agent.status == AgentLifecycleStatus.WAITING_FOR_INPUT:
            return f"{agent.name} is waiting for your input."
        if agent.status == AgentLifecycleStatus.FAILED:
            reason = agent.failure_reason or "No failure reason was recorded."
            return f"{agent.name} failed. {reason}"
        if agent.status == AgentLifecycleStatus.COMPLETED:
            result = agent.result_summary or agent.purpose
            return f"{agent.name} completed: {result}"
        step = agent.current_step or agent.purpose
        return f"{agent.name} is {agent.role.value}. Current work: {step}."

    def _severity(self, agent: AgentRecord) -> BriefingSeverity:
        """Return the severity implied by an agent's lifecycle status."""
        if agent.status == AgentLifecycleStatus.FAILED:
            return BriefingSeverity.ERROR
        if agent.status in self.WAITING_STATUSES:
            return BriefingSeverity.WARNING
        if agent.status == AgentLifecycleStatus.COMPLETED:
            return BriefingSeverity.SUCCESS
        return BriefingSeverity.INFO

    def _summary(
        self,
        active: list[AgentRecord],
        completed: list[AgentRecord],
        failed: list[AgentRecord],
        waiting: list[AgentRecord],
    ) -> str:
        """Summarize agent activity without inventing work."""
        if not active and not completed and not failed:
            return "No agents have run in this period."

        parts: list[str] = []
        if active:
            parts.append(f"{len(active)} active")
        if completed:
            parts.append(f"{len(completed)} completed")
        if failed:
            parts.append(f"{len(failed)} failed")
        summary = f"{', '.join(parts)}."
        if waiting:
            names = ", ".join(agent.name for agent in waiting)
            summary += f" Waiting on you: {names}."
        return summary


class MessagesCollector(BriefingCollector):
    """Report messages -- but only from a real, configured mail connector.

    The repository ships a mock Gmail connector that returns invented sample
    messages. Reading those aloud would be a fabrication, so a mock connector is
    reported as "not configured". This collector starts working on its own the
    day a connector with ``real_connector = True`` is registered.
    """

    section_type = BriefingSectionType.MESSAGES
    title = "Messages"
    source_id = "messages"

    def __init__(
        self,
        integration_registry: IntegrationRegistry,
        safe_logging_filter: SafeLoggingFilter | None = None,
    ) -> None:
        """Create the collector over the integration registry."""
        self.integration_registry = integration_registry
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect messages from a real mail connector, if one exists."""
        connector = self.integration_registry.find_real_connector("email")
        if connector is None:
            return self.unavailable(
                "No mail connector is configured. Jarvis cannot see your messages.",
                action_target="/settings?tab=integrations",
            )

        preview = connector.preview("read_email_preview")
        items = [
            BriefingItem(
                id=f"message_{index}",
                title=self.safe_logging_filter.sanitize_message(str(entry.get("subject", "Message"))),
                # Only a safe summary is carried. Full bodies stay in the mail
                # client and are never spoken or persisted.
                summary=self.safe_logging_filter.sanitize_message(
                    str(entry.get("safe_summary", "A message is waiting."))
                ),
                type=BriefingItemType.MESSAGE,
                severity=self._severity(str(entry.get("priority", "informational"))),
                source=connector.name,
                occurred_at=period_end,
                action_label="Open messages",
                action_target="/activity?tab=messages",
                safe_metadata={"priority": str(entry.get("priority", "informational"))},
            )
            for index, entry in enumerate(preview.get("items", []))
            if isinstance(entry, dict)
        ]

        urgent = [item for item in items if item.severity == BriefingSeverity.WARNING]
        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=(
                f"{len(items)} message{'' if len(items) == 1 else 's'}, "
                f"{len(urgent)} marked urgent."
                if items
                else "No new messages."
            ),
            items=items,
            severity=BriefingSeverity.WARNING if urgent else BriefingSeverity.INFO,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name=connector.name,
                    source_type="email",
                    availability=SourceAvailability.AVAILABLE,
                    trust_level=connector.trust_level,
                    retrieved_at=utc_now(),
                    mock_source=False,
                )
            ],
        )

    def _severity(self, priority: str) -> BriefingSeverity:
        """Map a connector's declared priority onto briefing severity."""
        if priority.lower() in {"urgent", "important"}:
            return BriefingSeverity.WARNING
        return BriefingSeverity.INFO


class ScheduleCollector(BriefingCollector):
    """Report calendar events -- but only from a real, configured calendar."""

    section_type = BriefingSectionType.SCHEDULE
    title = "Calendar and schedule"
    source_id = "calendar"

    def __init__(self, integration_registry: IntegrationRegistry) -> None:
        """Create the collector over the integration registry."""
        self.integration_registry = integration_registry

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect calendar events from a real calendar connector, if one exists."""
        connector = self.integration_registry.find_real_connector("calendar")
        if connector is None:
            # This is the distinction the whole briefing turns on: Jarvis says it
            # cannot see the calendar, not that the calendar is empty.
            return self.unavailable(
                "Calendar access is not configured.",
                action_target="/settings?tab=integrations",
            )

        preview = connector.preview("read_calendar_preview")
        items = [
            BriefingItem(
                id=f"event_{index}",
                title=str(entry.get("title", "Event")),
                summary=str(entry.get("summary", "No detail recorded.")),
                type=BriefingItemType.EVENT,
                severity=BriefingSeverity.INFO,
                source=connector.name,
                occurred_at=period_end,
                action_label="Open schedule",
                action_target="/activity?tab=schedule",
            )
            for index, entry in enumerate(preview.get("items", []))
            if isinstance(entry, dict)
        ]

        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=(
                f"{len(items)} event{'' if len(items) == 1 else 's'} scheduled."
                if items
                else "No events are scheduled."
            ),
            items=items,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name=connector.name,
                    source_type="calendar",
                    availability=SourceAvailability.AVAILABLE,
                    trust_level=connector.trust_level,
                    retrieved_at=utc_now(),
                )
            ],
        )


class WorldIntelligenceCollector(BriefingCollector):
    """Report world intelligence -- only from a configured trusted world source.

    No weather, news, finance, or emergency provider exists in this system yet.
    Rather than fake one, this section reports itself unavailable and stays that
    way until a real source is registered in the ``SourceRegistry``.

    Documentation connectors (official docs, GitHub docs) are deliberately *not*
    reused as world-news sources -- they answer a different question. Cyber
    advisories may appear, but only under an explicit Security Advisories label
    and only when that connector has been explicitly enabled.

    This collector opens no network connections. It reads registry state only.
    """

    section_type = BriefingSectionType.WORLD
    title = "World intelligence"
    source_id = "world_intelligence"

    WORLD_SOURCE_TYPES = {"news", "world_feed", "api"}

    def __init__(
        self,
        source_registry: SourceRegistry,
        retrieval_registry: RetrievalRegistry,
        security_advisories_enabled: bool = False,
    ) -> None:
        """Create the collector over the source and retrieval registries."""
        self.source_registry = source_registry
        self.retrieval_registry = retrieval_registry
        self.security_advisories_enabled = security_advisories_enabled

    def configured_world_sources(self) -> list:
        """Return enabled, trusted sources that can genuinely report on the world."""
        return [
            source
            for source in self.source_registry.get_all_sources()
            if source.source_type in self.WORLD_SOURCE_TYPES
            and source.enabled
            and source.trust_level in {"trusted", "partially_trusted"}
        ]

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect world intelligence, or report honestly that there is none."""
        sources = self.configured_world_sources()
        if not sources:
            section = self.unavailable(
                "No trusted world intelligence source is configured.",
                action_target="/settings?tab=sources",
            )
            section.source_references.append(
                SourceReference(
                    source_id="security_advisories",
                    name="Security advisories",
                    source_type="advisory",
                    availability=(
                        SourceAvailability.AVAILABLE
                        if self.security_advisories_enabled
                        else SourceAvailability.NOT_CONFIGURED
                    ),
                    detail=(
                        "Cyber advisory retrieval is enabled."
                        if self.security_advisories_enabled
                        else "Cyber advisory retrieval is not enabled."
                    ),
                )
            )
            section.summary = (
                "No trusted world intelligence source is configured. "
                "Jarvis has no weather, news, finance, or emergency provider and will not guess."
            )
            return section

        # A real source is registered. Its evidence still passes SourceTrust and
        # EvidenceVerifier before it can appear here; until a driver is wired for
        # it, Jarvis reports the source as registered but not yet readable rather
        # than inventing its contents.
        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=(
                f"{len(sources)} world source"
                f"{' is' if len(sources) == 1 else 's are'} registered "
                "but no retrieval driver is wired for them yet."
            ),
            items=[],
            available=False,
            unavailable_reason=(
                "A trusted world source is registered, but no retrieval driver can read it yet."
            ),
            severity=BriefingSeverity.INFO,
            source_references=[
                SourceReference(
                    source_id=source.source_id,
                    name=source.name,
                    source_type=source.source_type,
                    availability=SourceAvailability.UNAVAILABLE,
                    trust_level=source.trust_level,
                    evidence_status=EvidenceStanding.UNVERIFIED,
                    url=source.url,
                    detail="Registered source. No retrieval driver is wired.",
                )
                for source in sources
            ],
        )


class ApprovalsCollector(BriefingCollector):
    """Report what is genuinely waiting on a human decision."""

    section_type = BriefingSectionType.APPROVALS
    title = "Approvals"
    source_id = "approval_manager"

    def __init__(self, approval_manager: ApprovalManager) -> None:
        """Create the collector over the approval manager."""
        self.approval_manager = approval_manager

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect pending, expired, and blocked approvals."""
        pending = self.approval_manager.get_pending_approvals()
        expired = [
            approval
            for approval in self._all_approvals()
            if approval.status == ApprovalStatus.EXPIRED
        ]

        items = [
            BriefingItem(
                id=f"approval_{approval.approval_id}",
                title=approval.action_summary,
                summary=str(
                    approval.details.get("reason", "This action needs your decision.")
                ),
                type=BriefingItemType.APPROVAL,
                severity=self._severity(approval.risk_level),
                source=self.source_id,
                occurred_at=approval.created_at,
                action_label="Review approval",
                action_target=f"/activity?tab=approvals&approval={approval.approval_id}",
                approval_required=True,
                safe_metadata={
                    "risk_level": approval.risk_level.value,
                    "status": approval.status.value,
                },
            )
            for approval in [*pending, *expired]
        ]

        blocked = [
            approval
            for approval in [*pending, *expired]
            if approval.risk_level == RiskLevel.BLOCKED
        ]

        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=self._summary(len(pending), len(expired), len(blocked)),
            items=items,
            severity=BriefingSeverity.WARNING if pending else BriefingSeverity.INFO,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name="Approval manager",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                )
            ],
        )

    def _all_approvals(self) -> list:
        """Return every approval the manager currently holds."""
        return list(getattr(self.approval_manager, "_approvals", {}).values())

    def _severity(self, risk_level: RiskLevel) -> BriefingSeverity:
        """Map risk onto briefing severity."""
        if risk_level == RiskLevel.BLOCKED:
            return BriefingSeverity.CRITICAL
        if risk_level in {RiskLevel.HIGH, RiskLevel.CRITICAL}:
            return BriefingSeverity.ERROR
        if risk_level == RiskLevel.MEDIUM:
            return BriefingSeverity.WARNING
        return BriefingSeverity.INFO

    def _summary(self, pending: int, expired: int, blocked: int) -> str:
        """Summarize the approval queue."""
        if pending == 0 and expired == 0:
            return "Nothing is waiting for your approval."
        parts = []
        if pending:
            parts.append(f"{pending} approval{'' if pending == 1 else 's'} waiting")
        if expired:
            parts.append(f"{expired} expired")
        if blocked:
            parts.append(f"{blocked} blocked by policy")
        return f"{', '.join(parts)}."


class SkillsCollector(BriefingCollector):
    """Report the real state of Jarvis's skills."""

    section_type = BriefingSectionType.SKILLS
    title = "Skill intelligence"
    source_id = "skill_registry"

    def __init__(self, skill_registry: SkillRegistry) -> None:
        """Create the collector over the skill registry."""
        self.skill_registry = skill_registry

    def collect(self, period_start: datetime, period_end: datetime) -> BriefingSection:
        """Collect learned, in-flight, and recommended skills."""
        skills = self.skill_registry.list_skills()
        learned = [skill for skill in skills if skill.status == SkillStatus.LEARNED]
        evaluating = [
            skill
            for skill in skills
            if skill.status in {SkillStatus.LEARNING, SkillStatus.EVALUATING}
        ]
        recommended = [skill for skill in skills if skill.status == SkillStatus.RECOMMENDED]
        needs_approval = [
            skill for skill in skills if skill.status == SkillStatus.APPROVAL_REQUIRED
        ]

        items = [
            BriefingItem(
                id=f"skill_{skill.skill_id}",
                title=skill.name,
                summary=self._summarize_skill(skill),
                type=BriefingItemType.SKILL,
                severity=(
                    BriefingSeverity.WARNING
                    if skill.status == SkillStatus.APPROVAL_REQUIRED
                    else BriefingSeverity.INFO
                ),
                source=skill.source or self.source_id,
                action_label="Review skill",
                action_target=f"/settings?tab=skills&skill={skill.skill_id}",
                approval_required=skill.approval_required,
                safe_metadata={"status": skill.status.value},
            )
            for skill in [*needs_approval, *evaluating, *recommended, *learned]
        ]

        return BriefingSection(
            type=self.section_type,
            title=self.title,
            summary=self._summary(learned, evaluating, recommended, needs_approval),
            items=items,
            skills=[*needs_approval, *evaluating, *recommended, *learned],
            severity=BriefingSeverity.WARNING if needs_approval else BriefingSeverity.INFO,
            available=True,
            source_references=[
                SourceReference(
                    source_id=self.source_id,
                    name="Skill registry",
                    source_type="internal",
                    availability=SourceAvailability.AVAILABLE,
                    retrieved_at=utc_now(),
                ),
                SourceReference(
                    source_id="skill_catalog",
                    name="Reviewed skill catalog",
                    source_type="internal",
                    availability=(
                        SourceAvailability.AVAILABLE
                        if self.skill_registry.catalog_configured()
                        else SourceAvailability.NOT_CONFIGURED
                    ),
                    detail=(
                        "Recommendations come from the reviewed local catalog."
                        if self.skill_registry.catalog_configured()
                        else "No reviewed skill catalog is configured, so Jarvis has nothing to recommend."
                    ),
                ),
            ],
        )

    def _summarize_skill(self, skill) -> str:
        """Return a status-appropriate summary for one skill."""
        if skill.status == SkillStatus.RECOMMENDED:
            return skill.reason_recommended or skill.purpose
        if skill.status == SkillStatus.APPROVAL_REQUIRED:
            return f"Waiting for your approval. {skill.purpose}"
        if skill.status in {SkillStatus.LEARNING, SkillStatus.EVALUATING}:
            step = skill.evaluation_step or "review in progress"
            return f"Currently at: {step.replace('_', ' ')}. {skill.purpose}"
        return skill.purpose

    def _summary(
        self,
        learned: list,
        evaluating: list,
        recommended: list,
        needs_approval: list,
    ) -> str:
        """Summarize skill state without overclaiming."""
        parts: list[str] = []
        if learned:
            parts.append(f"{len(learned)} learned")
        if evaluating:
            parts.append(f"{len(evaluating)} in review")
        if recommended:
            parts.append(f"{len(recommended)} recommended")
        if needs_approval:
            parts.append(f"{len(needs_approval)} awaiting approval")
        if not parts:
            return "No skills have been learned, and none are recommended."
        return f"{', '.join(parts)}."
