from jarvis_brain.agents.agent_naming_service import AgentNamingService
from jarvis_platform.schemas.agent_lifecycle import AgentRole
import pytest


def test_role_based_names_generated() -> None:
    service = AgentNamingService()

    assert service.generate_name(AgentRole.PLANNER, 1) == "Forge-01"
    assert service.generate_name(AgentRole.SECURITY, 2) == "Shield-02"
    assert service.generate_name(AgentRole.VISION, 3) == "Optic-03"


def test_sequence_formatting_works() -> None:
    service = AgentNamingService()

    assert service.generate_name(AgentRole.CODER, 12) == "Builder-12"


def test_unknown_role_fallback_works() -> None:
    service = AgentNamingService()

    assert service.generate_name(AgentRole.UNKNOWN, 1) == "Nova-01"


def test_name_pool_rotates_deterministically() -> None:
    service = AgentNamingService()

    assert service.get_base_name(AgentRole.PLANNER, 4) == "Forge"
    assert service.generate_name(AgentRole.PLANNER, 4) == "Forge-04"


def test_sequence_must_be_at_least_one() -> None:
    with pytest.raises(ValueError):
        AgentNamingService().generate_name(AgentRole.PLANNER, 0)
