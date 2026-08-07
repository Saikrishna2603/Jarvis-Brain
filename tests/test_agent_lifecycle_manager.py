from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_platform.schemas.agent_lifecycle import (
    AgentEventType,
    AgentLifecycleStatus,
    AgentRole,
)
from jarvis_platform.schemas.agent_stream import AgentStreamEnvelope, AgentStreamEventType
from unittest.mock import MagicMock


def test_create_agent() -> None:
    manager = AgentLifecycleManager()

    agent = manager.create_agent(AgentRole.PLANNER, "Plan a safe workflow")

    assert agent.role == AgentRole.PLANNER
    assert agent.status == AgentLifecycleStatus.CREATED
    assert agent.agent_id in manager.active_agents


def test_update_status_creates_event() -> None:
    manager = AgentLifecycleManager()
    agent = manager.create_agent(AgentRole.CODER, "Inspect code")

    updated = manager.update_status(
        agent.agent_id,
        AgentLifecycleStatus.WORKING,
        current_step="Reading files",
        progress_percent=35,
    )

    assert updated.status == AgentLifecycleStatus.WORKING
    assert updated.progress_percent == 35
    assert manager.events[-1].event_type == AgentEventType.STATUS_CHANGED


def test_assign_task_updates_status() -> None:
    manager = AgentLifecycleManager()
    agent = manager.create_agent(AgentRole.CODER, "Inspect code")

    assigned = manager.assign_task(agent.agent_id, "Inspect routing code")

    assert assigned.status == AgentLifecycleStatus.ASSIGNED
    assert assigned.current_step == "Inspect routing code"
    assert manager.events[-1].event_type == AgentEventType.TASK_ASSIGNED


def test_complete_agent_moves_to_completed() -> None:
    manager = AgentLifecycleManager()
    agent = manager.create_agent(AgentRole.REVIEWER, "Review output")

    completed = manager.complete_agent(agent.agent_id, "Review complete")

    assert completed.status == AgentLifecycleStatus.COMPLETED
    assert agent.agent_id not in manager.active_agents
    assert agent.agent_id in manager.completed_agents


def test_fail_agent_moves_to_failed() -> None:
    manager = AgentLifecycleManager()
    agent = manager.create_agent(AgentRole.SECURITY, "Check safety")

    failed = manager.fail_agent(agent.agent_id, "Blocked unsafe request")

    assert failed.status == AgentLifecycleStatus.FAILED
    assert agent.agent_id in manager.failed_agents


def test_terminate_and_archive_agents() -> None:
    manager = AgentLifecycleManager()
    terminated_agent = manager.create_agent(AgentRole.EXECUTOR, "Demo work")
    archived_agent = manager.create_agent(AgentRole.MEMORY, "Store summary")

    terminated = manager.terminate_agent(terminated_agent.agent_id, "Stopped safely")
    archived = manager.archive_agent(archived_agent.agent_id)
    snapshot = manager.get_snapshot()

    assert terminated.status == AgentLifecycleStatus.TERMINATED
    assert archived.status == AgentLifecycleStatus.ARCHIVED
    assert archived in snapshot.archived_agents
    assert manager.events[-1].event_type == AgentEventType.AGENT_ARCHIVED


def test_invalid_progress_is_rejected() -> None:
    manager = AgentLifecycleManager()
    agent = manager.create_agent(AgentRole.PLANNER, "Plan")

    try:
        manager.update_status(agent.agent_id, AgentLifecycleStatus.WORKING, progress_percent=101)
    except ValueError as error:
        assert "between 0 and 100" in str(error)
    else:
        raise AssertionError("Expected invalid progress to fail")


def test_snapshot_contains_agents_events_and_graph() -> None:
    manager = AgentLifecycleManager()
    parent = manager.create_agent(AgentRole.PLANNER, "Coordinate")
    child = manager.create_agent(
        AgentRole.RESEARCHER,
        "Research",
        parent_agent_id=parent.agent_id,
    )

    snapshot = manager.get_snapshot()

    assert len(snapshot.active_agents) == 2
    assert snapshot.recent_events
    assert any(node.id == child.agent_id for node in snapshot.graph_nodes)
    assert any(edge.target == child.agent_id for edge in snapshot.graph_edges)


def test_demo_snapshot_is_marked_demo() -> None:
    manager = AgentLifecycleManager()

    snapshot = manager.create_demo_snapshot()

    assert snapshot.metadata["demo"] is True
    assert snapshot.active_agents or snapshot.completed_agents
    assert all(
        agent.metadata.get("demo")
        for agent in [*snapshot.active_agents, *snapshot.completed_agents, *snapshot.failed_agents]
    )
    assert all(event.metadata.get("demo") for event in snapshot.recent_events)


def test_reset_demo_preserves_real_agents() -> None:
    manager = AgentLifecycleManager()
    real_agent = manager.create_agent(AgentRole.SECURITY, "Review a real request")
    manager.create_demo_snapshot()

    snapshot = manager.reset_demo_data()

    assert manager.get_agent(real_agent.agent_id) == real_agent
    assert snapshot.metadata["demo"] is False
    assert all(not agent.metadata.get("demo") for agent in snapshot.active_agents)


def test_secret_values_are_redacted() -> None:
    manager = AgentLifecycleManager()
    secret = "sk-test123456789012345678901234567890"

    agent = manager.create_agent(AgentRole.SECURITY, f"Review {secret}")
    manager.assign_task(agent.agent_id, f"Inspect token={secret}")

    serialized = str(manager.get_snapshot().model_dump())
    assert secret not in serialized


def _recording_broadcaster() -> MagicMock:
    broadcaster = MagicMock()
    broadcaster.create_envelope.side_effect = lambda event_type, **kwargs: AgentStreamEnvelope(
        stream_event_id=f"stream-{event_type.value}",
        event_type=event_type,
        agent_id=kwargs.get("agent").agent_id if kwargs.get("agent") else None,
        data=kwargs.get("data") or {},
        demo=kwargs.get("demo", False),
    )
    return broadcaster


def test_manager_publishes_create_status_complete_and_failure() -> None:
    broadcaster = _recording_broadcaster()
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)
    completed_agent = manager.create_agent(AgentRole.PLANNER, "Plan")
    manager.update_status(completed_agent.agent_id, AgentLifecycleStatus.WORKING)
    manager.complete_agent(completed_agent.agent_id, "Done")
    failed_agent = manager.create_agent(AgentRole.CODER, "Inspect")
    manager.fail_agent(failed_agent.agent_id, "Failed safely")

    event_types = [
        call.args[0].event_type
        for call in broadcaster.publish.call_args_list
    ]
    assert AgentStreamEventType.AGENT_CREATED in event_types
    assert AgentStreamEventType.AGENT_UPDATED in event_types
    assert AgentStreamEventType.AGENT_COMPLETED in event_types
    assert AgentStreamEventType.AGENT_FAILED in event_types
    assert broadcaster.publish_lifecycle_event.called


def test_demo_stream_publications_are_marked_demo() -> None:
    broadcaster = _recording_broadcaster()
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)

    manager.create_demo_snapshot()

    envelopes = [call.args[0] for call in broadcaster.publish.call_args_list]
    assert any(
        envelope.event_type == AgentStreamEventType.DEMO_STARTED
        and envelope.demo
        for envelope in envelopes
    )
    assert any(envelope.demo for envelope in envelopes)


def test_broadcaster_exception_does_not_break_lifecycle_operation() -> None:
    broadcaster = MagicMock()
    broadcaster.publish_lifecycle_event.side_effect = RuntimeError("stream failed")
    broadcaster.create_envelope.side_effect = RuntimeError("stream failed")
    manager = AgentLifecycleManager(event_broadcaster=broadcaster)

    agent = manager.create_agent(AgentRole.SECURITY, "Review safely")
    updated = manager.update_status(agent.agent_id, AgentLifecycleStatus.REVIEWING)

    assert updated.status == AgentLifecycleStatus.REVIEWING
