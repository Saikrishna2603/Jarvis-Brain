from fastapi import APIRouter, HTTPException

from jarvis_brain.engine.brain_engine import BrainEngine
from jarvis_platform.schemas.audit_log import AuditLog
from jarvis_platform.schemas.brain_request import BrainRequest
from jarvis_platform.schemas.brain_orchestration import BrainOrchestrationResult
from jarvis_platform.schemas.brain_response import BrainResponse
from jarvis_platform.schemas.intent_result import IntentResult
from jarvis_platform.schemas.plan import ExecutionPlan
from jarvis_platform.schemas.task_memory import TaskMemory


router = APIRouter()
brain_engine = BrainEngine()


@router.post("/brain/think", response_model=BrainResponse)
def think(request: BrainRequest) -> BrainResponse:
    """Process one user request through the shared brain engine."""
    raw_input = request.raw_input if request.raw_input is not None else request.content
    return brain_engine.process_input(raw_input)


@router.post("/brain/intent", response_model=IntentResult)
def resolve_intent(request: BrainRequest, use_llm: bool = False) -> IntentResult:
    """Resolve user input without executing actions or creating memory."""
    raw_input = request.raw_input if request.raw_input is not None else request.content
    if use_llm:
        return brain_engine.llm_assisted_intent_resolver.resolve(
            raw_input,
            context=request.context,
        )
    return brain_engine.intent_resolver.resolve(raw_input)


@router.get("/brain/orchestrator/status")
def brain_orchestrator_status() -> dict[str, object]:
    """Return safe Brain Orchestrator status."""
    return {
        "enabled": brain_engine.brain_orchestrator.enabled,
        "mode": "llm_first_with_deterministic_fallback",
        "execution_control": False,
        "tools_execute_directly": False,
        "safety_authority": [
            "SecretGuard",
            "InputSecurityGateway",
            "RiskClassifier",
            "PermissionPolicyEngine",
            "ActionFirewall",
            "ApprovalManager",
        ],
    }


@router.post("/brain/orchestrator/preview", response_model=BrainOrchestrationResult)
def brain_orchestrator_preview(request: BrainRequest) -> BrainOrchestrationResult:
    """Preview orchestration without executing tools or creating task memory."""
    raw_input = request.raw_input if request.raw_input is not None else request.content
    return brain_engine.brain_orchestrator.orchestrate(
        raw_input=raw_input,
        metadata=request.context,
    )


@router.get("/brain/audit", response_model=list[AuditLog])
def get_audit_events(event_type: str | None = None) -> list[AuditLog]:
    """Return audit events recorded by the shared brain engine."""
    if event_type is not None:
        return brain_engine.audit_manager.get_events_by_type(event_type)

    return brain_engine.audit_manager.get_all_events()


@router.get("/brain/tasks", response_model=list[TaskMemory])
def get_tasks(action: str | None = None) -> list[TaskMemory]:
    """Return task memories recorded by the shared brain engine."""
    if action is not None:
        return brain_engine.task_memory_manager.get_tasks_by_action(action)

    return brain_engine.task_memory_manager.get_all_tasks()


@router.get("/brain/plans", response_model=list[ExecutionPlan])
def get_plans(status: str | None = None) -> list[ExecutionPlan]:
    """Return execution plans recorded by the shared brain engine."""
    if status is not None:
        return brain_engine.plan_memory_manager.get_plans_by_status(status)

    return brain_engine.plan_memory_manager.get_all_plans()


@router.get("/brain/plans/{plan_id}", response_model=ExecutionPlan)
def get_plan(plan_id: str) -> ExecutionPlan:
    """Return one execution plan by id."""
    plan = brain_engine.plan_memory_manager.get_plan(plan_id)
    if plan is None:
        raise HTTPException(status_code=404, detail=f"Plan not found: {plan_id}")

    return plan
