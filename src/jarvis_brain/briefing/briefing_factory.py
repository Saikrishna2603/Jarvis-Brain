from jarvis_brain.agents.agent_lifecycle_manager import AgentLifecycleManager
from jarvis_brain.briefing.briefing_store import BriefingStore
from jarvis_brain.briefing.collectors import (
    ActivityCollector,
    AgentActivityCollector,
    ApprovalsCollector,
    MessagesCollector,
    ScheduleCollector,
    SkillsCollector,
    SystemReadinessCollector,
    WorldIntelligenceCollector,
)
from jarvis_brain.briefing.daily_briefing_service import DailyBriefingService
from jarvis_brain.briefing.system_readiness import build_system_readiness_provider
from jarvis_brain.engine.brain_engine import BrainEngine
from jarvis_platform.integrations.integration_registry import IntegrationRegistry
from jarvis_brain.ports import SourceRegistry
from jarvis_brain.ports import RetrievalRegistry
from jarvis_brain.ports import create_default_retrieval_registry
from jarvis_brain.ports import SkillRegistry


def create_daily_briefing_service(
    brain_engine: BrainEngine,
    agent_lifecycle_manager: AgentLifecycleManager,
    skill_registry: SkillRegistry,
    integration_registry: IntegrationRegistry | None = None,
    retrieval_registry: RetrievalRegistry | None = None,
    source_registry: SourceRegistry | None = None,
    briefing_store: BriefingStore | None = None,
) -> DailyBriefingService:
    """Wire the daily briefing service to the live Jarvis subsystems.

    Every collector here reads a system that already exists. Nothing is stubbed:
    the sections that have no real backing source (messages, calendar, world)
    report themselves unavailable rather than being filled with sample data.
    """
    integrations = integration_registry or IntegrationRegistry()
    retrieval = retrieval_registry or create_default_retrieval_registry()
    sources = source_registry or SourceRegistry()

    collectors = [
        SystemReadinessCollector(
            status_provider=build_system_readiness_provider(integrations, retrieval),
        ),
        ActivityCollector(
            task_memory_manager=brain_engine.task_memory_manager,
            audit_manager=brain_engine.audit_manager,
        ),
        AgentActivityCollector(lifecycle_manager=agent_lifecycle_manager),
        MessagesCollector(integration_registry=integrations),
        ScheduleCollector(integration_registry=integrations),
        WorldIntelligenceCollector(
            source_registry=sources,
            retrieval_registry=retrieval,
        ),
        ApprovalsCollector(approval_manager=brain_engine.approval_manager),
        SkillsCollector(skill_registry=skill_registry),
    ]

    return DailyBriefingService(
        collectors=collectors,
        briefing_store=briefing_store or BriefingStore(),
    )
