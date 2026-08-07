from collections import defaultdict
from typing import Any
from uuid import uuid4

from jarvis_brain.agents.agent_event_broadcaster import AgentEventBroadcaster
from jarvis_brain.agents.agent_naming_service import AgentNamingService
from jarvis_platform.schemas.agent_lifecycle import (
    AgentEvent,
    AgentEventType,
    AgentGraphEdge,
    AgentGraphNode,
    AgentLifecycleSnapshot,
    AgentLifecycleStatus,
    AgentRecord,
    AgentRole,
)
from jarvis_platform.schemas.common import utc_now
from jarvis_platform.schemas.agent_stream import AgentStreamEventType
from jarvis_platform.security.safe_logging_filter import SafeLoggingFilter


class AgentLifecycleManager:
    """Track current and recent Jarvis agent lifecycle activity in memory."""

    def __init__(
        self,
        naming_service: AgentNamingService | None = None,
        safe_logging_filter: SafeLoggingFilter | None = None,
        max_recent_events: int = 200,
        event_broadcaster: AgentEventBroadcaster | None = None,
    ) -> None:
        """Create an empty lifecycle manager."""
        self.naming_service = naming_service or AgentNamingService()
        self.safe_logging_filter = safe_logging_filter or SafeLoggingFilter()
        self.max_recent_events = max(1, max_recent_events)
        self.event_broadcaster = event_broadcaster
        self.active_agents: dict[str, AgentRecord] = {}
        self.completed_agents: dict[str, AgentRecord] = {}
        self.failed_agents: dict[str, AgentRecord] = {}
        self.archived_agents: dict[str, AgentRecord] = {}
        self.events: list[AgentEvent] = []
        self.role_sequences: defaultdict[AgentRole, int] = defaultdict(int)

    def create_agent(
        self,
        role: AgentRole | str,
        purpose: str,
        request_id: str | None = None,
        parent_agent_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRecord:
        """Create and register a new active agent."""
        agent_role = self._coerce_role(role)
        self.role_sequences[agent_role] += 1
        agent = AgentRecord(
            agent_id=f"agent_{uuid4().hex}",
            name=self.naming_service.generate_name(
                agent_role,
                self.role_sequences[agent_role],
            ),
            role=agent_role,
            purpose=self.safe_logging_filter.sanitize_message(purpose),
            status=AgentLifecycleStatus.CREATED,
            request_id=request_id,
            parent_agent_id=parent_agent_id,
            metadata=self.safe_logging_filter.sanitize_metadata(metadata or {}),
        )
        self.active_agents[agent.agent_id] = agent
        self.add_event(
            agent.agent_id,
            AgentEventType.AGENT_CREATED,
            f"{agent.name} was created.",
            status=agent.status,
            metadata=agent.metadata,
        )
        self.add_event(
            agent.agent_id,
            AgentEventType.NAME_ASSIGNED,
            f"Jarvis named the agent {agent.name}.",
            status=agent.status,
            metadata=agent.metadata,
        )
        self.add_event(
            agent.agent_id,
            AgentEventType.ROLE_ASSIGNED,
            f"{agent.name} was assigned the {agent.role.value} role.",
            status=agent.status,
            metadata=agent.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_CREATED, agent)
        return agent

    def assign_task(self, agent_id: str, task: str) -> AgentRecord:
        """Assign descriptive work without executing it."""
        agent = self._get_agent_or_raise(agent_id)
        safe_task = self.safe_logging_filter.sanitize_message(task)
        updated = agent.model_copy(
            update={
                "status": AgentLifecycleStatus.ASSIGNED,
                "current_step": safe_task,
                "updated_at": utc_now(),
            }
        )
        self._store_agent(updated)
        self.add_event(
            agent_id,
            AgentEventType.TASK_ASSIGNED,
            f"{updated.name} was assigned: {safe_task}",
            status=updated.status,
            metadata=updated.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_UPDATED, updated)
        return updated

    def update_status(
        self,
        agent_id: str,
        status: AgentLifecycleStatus | str,
        current_step: str | None = None,
        progress_percent: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentRecord:
        """Update an agent status and create a status-change event."""
        agent = self._get_agent_or_raise(agent_id)
        safe_status = self._coerce_status(status)
        updates: dict[str, Any] = {
            "status": safe_status,
            "updated_at": utc_now(),
        }
        if current_step is not None:
            updates["current_step"] = self.safe_logging_filter.sanitize_message(
                current_step
            )
        if progress_percent is not None:
            if not 0 <= progress_percent <= 100:
                raise ValueError("Agent progress must be between 0 and 100.")
            updates["progress_percent"] = progress_percent
        if metadata:
            updates["metadata"] = {
                **agent.metadata,
                **self.safe_logging_filter.sanitize_metadata(metadata),
            }

        updated = agent.model_copy(update=updates)
        self._store_agent(updated)
        self.add_event(
            agent_id,
            AgentEventType.STATUS_CHANGED,
            f"{updated.name} is now {updated.status.value}.",
            status=updated.status,
            metadata={**updated.metadata, "current_step": updated.current_step},
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_UPDATED, updated)
        return updated

    def add_event(
        self,
        agent_id: str,
        event_type: AgentEventType | str,
        message: str,
        status: AgentLifecycleStatus | str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> AgentEvent:
        """Record a sanitized lifecycle event."""
        event = AgentEvent(
            event_id=f"agent_event_{uuid4().hex}",
            agent_id=agent_id,
            event_type=self._coerce_event_type(event_type),
            message=self.safe_logging_filter.sanitize_message(message),
            status=self._coerce_status(status) if status is not None else None,
            metadata=self.safe_logging_filter.sanitize_metadata(metadata or {}),
        )
        self.events.append(event)
        self.events = self.events[-self.max_recent_events :]
        try:
            if self.event_broadcaster is not None:
                self.event_broadcaster.publish_lifecycle_event(
                    event, self._find_agent(agent_id)
                )
        except Exception:
            pass
        return event

    def complete_agent(
        self,
        agent_id: str,
        result_summary: str | None = None,
    ) -> AgentRecord:
        """Move an agent to the completed list."""
        agent = self._get_agent_or_raise(agent_id)
        completed = agent.model_copy(
            update={
                "status": AgentLifecycleStatus.COMPLETED,
                "progress_percent": 100,
                "result_summary": self.safe_logging_filter.sanitize_message(
                    result_summary or "Agent completed."
                ),
                "updated_at": utc_now(),
                "ended_at": utc_now(),
            }
        )
        self.active_agents.pop(agent_id, None)
        self.failed_agents.pop(agent_id, None)
        self.completed_agents[agent_id] = completed
        if result_summary:
            self.add_event(
                agent_id,
                AgentEventType.RESULT_CREATED,
                completed.result_summary or "Agent result created.",
                status=completed.status,
                metadata=completed.metadata,
            )
        self.add_event(
            agent_id,
            AgentEventType.AGENT_COMPLETED,
            completed.result_summary or "Agent completed.",
            status=completed.status,
            metadata=completed.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_COMPLETED, completed)
        return completed

    def fail_agent(self, agent_id: str, failure_reason: str) -> AgentRecord:
        """Move an agent to the failed list."""
        agent = self._get_agent_or_raise(agent_id)
        failed = agent.model_copy(
            update={
                "status": AgentLifecycleStatus.FAILED,
                "failure_reason": self.safe_logging_filter.sanitize_message(
                    failure_reason
                ),
                "updated_at": utc_now(),
                "ended_at": utc_now(),
            }
        )
        self.active_agents.pop(agent_id, None)
        self.completed_agents.pop(agent_id, None)
        self.failed_agents[agent_id] = failed
        self.add_event(
            agent_id,
            AgentEventType.AGENT_FAILED,
            failed.failure_reason or "Agent failed.",
            status=failed.status,
            metadata=failed.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_FAILED, failed)
        return failed

    def terminate_agent(
        self, agent_id: str, reason: str | None = None
    ) -> AgentRecord:
        """Terminate an agent without executing any additional work."""
        agent = self._get_agent_or_raise(agent_id)
        safe_reason = self.safe_logging_filter.sanitize_message(
            reason or "Agent terminated."
        )
        terminated = agent.model_copy(
            update={
                "status": AgentLifecycleStatus.TERMINATED,
                "failure_reason": safe_reason,
                "updated_at": utc_now(),
                "ended_at": utc_now(),
            }
        )
        self.active_agents.pop(agent_id, None)
        self.completed_agents.pop(agent_id, None)
        self.archived_agents.pop(agent_id, None)
        self.failed_agents[agent_id] = terminated
        self.add_event(
            agent_id,
            AgentEventType.AGENT_TERMINATED,
            safe_reason,
            status=terminated.status,
            metadata=terminated.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_UPDATED, terminated)
        return terminated

    def archive_agent(self, agent_id: str) -> AgentRecord:
        """Mark an agent as archived while keeping it visible in history."""
        agent = self._get_agent_or_raise(agent_id)
        archived = agent.model_copy(
            update={
                "status": AgentLifecycleStatus.ARCHIVED,
                "updated_at": utc_now(),
                "ended_at": agent.ended_at or utc_now(),
            }
        )
        self.active_agents.pop(agent_id, None)
        self.completed_agents.pop(agent_id, None)
        self.failed_agents.pop(agent_id, None)
        self.archived_agents[agent_id] = archived
        self.add_event(
            agent_id,
            AgentEventType.AGENT_ARCHIVED,
            f"{archived.name} was archived.",
            status=archived.status,
            metadata=archived.metadata,
        )
        self._publish_agent_state(AgentStreamEventType.AGENT_ARCHIVED, archived)
        return archived

    def get_agent(self, agent_id: str) -> AgentRecord | None:
        """Return one agent from any lifecycle collection."""
        return self._find_agent(agent_id)

    def get_agent_events(self, agent_id: str) -> list[AgentEvent]:
        """Return one agent's events in chronological order."""
        return [event for event in self.events if event.agent_id == agent_id]

    def get_snapshot(self) -> AgentLifecycleSnapshot:
        """Return the current lifecycle snapshot."""
        all_agents = [
            *self.active_agents.values(),
            *self.completed_agents.values(),
            *self.failed_agents.values(),
            *self.archived_agents.values(),
        ]
        return AgentLifecycleSnapshot(
            active_agents=list(self.active_agents.values()),
            completed_agents=list(self.completed_agents.values()),
            failed_agents=list(self.failed_agents.values()),
            archived_agents=list(self.archived_agents.values()),
            recent_events=list(reversed(self.events[-25:])),
            graph_nodes=[
                AgentGraphNode(
                    id=agent.agent_id,
                    label=agent.name,
                    role=agent.role.value,
                    status=agent.status.value,
                    metadata=self.safe_logging_filter.sanitize_metadata(agent.metadata),
                )
                for agent in all_agents
            ],
            graph_edges=[
                AgentGraphEdge(
                    source=agent.parent_agent_id,
                    target=agent.agent_id,
                    label="delegated",
                    metadata={"demo": bool(agent.metadata.get("demo"))},
                )
                for agent in all_agents
                if agent.parent_agent_id
            ],
            metadata={
                "mode": "in_memory",
                "demo": any(agent.metadata.get("demo") for agent in all_agents),
            },
        )

    def reset_demo_data(self) -> AgentLifecycleSnapshot:
        """Remove demo records while preserving any real lifecycle state."""
        demo_ids = {
            agent.agent_id
            for collection in (
                self.active_agents,
                self.completed_agents,
                self.failed_agents,
                self.archived_agents,
            )
            for agent in collection.values()
            if agent.metadata.get("demo")
        }
        for collection in (
            self.active_agents,
            self.completed_agents,
            self.failed_agents,
            self.archived_agents,
        ):
            for agent_id in demo_ids:
                collection.pop(agent_id, None)
        self.events = [
            event
            for event in self.events
            if event.agent_id not in demo_ids and not event.metadata.get("demo")
        ]
        snapshot = self.get_snapshot()
        self._publish_envelope(
            AgentStreamEventType.DEMO_RESET,
            data=snapshot.model_dump(mode="json"),
            demo=True,
        )
        return snapshot

    def create_demo_snapshot(self) -> AgentLifecycleSnapshot:
        """Create clearly marked demo lifecycle activity for the dashboard."""
        self.reset_demo_data()
        planner = self.create_agent(
            AgentRole.PLANNER,
            "Demo: break a user request into safe specialist work.",
            request_id="demo-request",
            metadata={"demo": True},
        )
        researcher = self.create_agent(
            AgentRole.RESEARCHER,
            "Demo: collect trusted evidence for the request.",
            request_id="demo-request",
            parent_agent_id=planner.agent_id,
            metadata={"demo": True},
        )
        security = self.create_agent(
            AgentRole.SECURITY,
            "Demo: review the proposed action for safety.",
            request_id="demo-request",
            parent_agent_id=planner.agent_id,
            metadata={"demo": True},
        )
        integration = self.create_agent(
            AgentRole.INTEGRATION,
            "Demo: prepare an approval-gated connector preview.",
            request_id="demo-request",
            parent_agent_id=planner.agent_id,
            metadata={"demo": True},
        )
        coder = self.create_agent(
            AgentRole.CODER,
            "Demo: inspect a non-executing code proposal.",
            request_id="demo-request",
            parent_agent_id=planner.agent_id,
            metadata={"demo": True},
        )

        for agent in (planner, researcher, security, integration, coder):
            self.assign_task(agent.agent_id, agent.purpose)

        self.update_status(
            planner.agent_id,
            AgentLifecycleStatus.WORKING,
            current_step="Demo planning preview",
            progress_percent=60,
        )
        self.update_status(
            researcher.agent_id,
            AgentLifecycleStatus.COMPLETED,
            current_step="Demo evidence review complete",
            progress_percent=100,
        )
        self.complete_agent(
            researcher.agent_id,
            "Demo research summary completed; no live retrieval occurred.",
        )
        self.update_status(
            security.agent_id,
            AgentLifecycleStatus.REVIEWING,
            current_step="Demo safety review",
            progress_percent=80,
        )
        self.update_status(
            integration.agent_id,
            AgentLifecycleStatus.WAITING_FOR_APPROVAL,
            current_step="Demo approval required; no connector action executed",
            progress_percent=45,
        )
        self.update_status(
            coder.agent_id,
            AgentLifecycleStatus.WORKING,
            current_step="Demo code inspection",
            progress_percent=35,
        )
        self.fail_agent(
            coder.agent_id,
            "Demo failure: validation rejected the sample proposal.",
        )
        self.add_event(
            planner.agent_id,
            AgentEventType.MESSAGE_SENT,
            "Demo lifecycle event: planner coordinates with researcher.",
            status=AgentLifecycleStatus.COMMUNICATING,
            metadata={"demo": True},
        )
        snapshot = self.get_snapshot()
        self._publish_envelope(
            AgentStreamEventType.DEMO_STARTED,
            data=snapshot.model_dump(mode="json"),
            demo=True,
        )
        try:
            if self.event_broadcaster is not None:
                self.event_broadcaster.publish_snapshot(snapshot)
        except Exception:
            pass
        return snapshot

    def _publish_agent_state(
        self, event_type: AgentStreamEventType, agent: AgentRecord
    ) -> None:
        try:
            if self.event_broadcaster is not None:
                self.event_broadcaster.publish(
                    self.event_broadcaster.create_envelope(
                        event_type, agent=agent
                    )
                )
        except Exception:
            pass

    def _publish_envelope(
        self,
        event_type: AgentStreamEventType,
        *,
        data: dict[str, Any] | None = None,
        demo: bool = False,
    ) -> None:
        try:
            if self.event_broadcaster is not None:
                self.event_broadcaster.publish(
                    self.event_broadcaster.create_envelope(
                        event_type, data=data, demo=demo
                    )
                )
        except Exception:
            pass

    def _store_agent(self, agent: AgentRecord) -> None:
        if agent.agent_id in self.completed_agents:
            self.completed_agents[agent.agent_id] = agent
        elif agent.agent_id in self.failed_agents:
            self.failed_agents[agent.agent_id] = agent
        elif agent.agent_id in self.archived_agents:
            self.archived_agents[agent.agent_id] = agent
        else:
            self.active_agents[agent.agent_id] = agent

    def _get_agent_or_raise(self, agent_id: str) -> AgentRecord:
        agent = self._find_agent(agent_id)
        if agent is None:
            raise KeyError(f"Agent not found: {agent_id}")
        return agent

    def _find_agent(self, agent_id: str) -> AgentRecord | None:
        return (
            self.active_agents.get(agent_id)
            or self.completed_agents.get(agent_id)
            or self.failed_agents.get(agent_id)
            or self.archived_agents.get(agent_id)
        )

    def _coerce_role(self, role: AgentRole | str) -> AgentRole:
        try:
            return role if isinstance(role, AgentRole) else AgentRole(role)
        except ValueError:
            return AgentRole.UNKNOWN

    def _coerce_status(
        self, status: AgentLifecycleStatus | str
    ) -> AgentLifecycleStatus:
        return status if isinstance(status, AgentLifecycleStatus) else AgentLifecycleStatus(status)

    def _coerce_event_type(self, event_type: AgentEventType | str) -> AgentEventType:
        return (
            event_type
            if isinstance(event_type, AgentEventType)
            else AgentEventType(event_type)
        )
