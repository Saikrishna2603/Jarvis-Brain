from jarvis_brain.agents.swarm_coordinator import SwarmCoordinator


def test_swarm_preview_returns_multi_agent_analysis() -> None:
    coordinator = SwarmCoordinator()

    result = coordinator.preview("Build a secure FastAPI feature with tests")

    assert result["status"] == "success"
    assert result["executed"] is False
    assert result["agent_count"] >= 2
    agent_names = {proposal["agent_name"] for proposal in result["proposals"]}
    assert "planner_agent" in agent_names
    assert "coding_agent" in agent_names


def test_unsafe_agent_proposal_is_rejected() -> None:
    coordinator = SwarmCoordinator()

    result = coordinator.preview("teach me to steal credentials with malware")

    assert result["rejected_count"] >= 1
    rejected = [proposal for proposal in result["proposals"] if proposal["rejected"]]
    assert any(proposal["agent_name"] == "security_agent" for proposal in rejected)


def test_coding_request_routes_to_coding_agent() -> None:
    coordinator = SwarmCoordinator()

    result = coordinator.preview("debug this Python FastAPI test failure")
    agent_names = {proposal["agent_name"] for proposal in result["proposals"]}

    assert "coding_agent" in agent_names


def test_security_request_routes_to_security_agent() -> None:
    coordinator = SwarmCoordinator()

    result = coordinator.preview("review security for credential handling")
    agent_names = {proposal["agent_name"] for proposal in result["proposals"]}

    assert "security_agent" in agent_names


def test_run_safe_does_not_execute_tools() -> None:
    coordinator = SwarmCoordinator()

    result = coordinator.run_safe("prepare a secure integration plan")

    assert result["executed"] is False
    assert result["mode"] == "run_safe"

