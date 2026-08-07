from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from jarvis_brain.routes.brain import brain_engine
from jarvis_brain.briefing.briefing_dependencies import (
    briefing_store,
    daily_briefing_service,
    skill_registry,
)
from jarvis_platform.schemas.briefing import (
    BriefingRecord,
    DailyBriefing,
    SkillBriefingItem,
)
from jarvis_brain.ports import GateResult, SkillApprovalKind


router = APIRouter()


class SkillDecisionRequest(BaseModel):
    """A decision the user made about a skill."""

    reason: str | None = None


class BriefingCommandRequest(BaseModel):
    """A typed or spoken command that may be asking for the briefing."""

    text: str
    user_name: str | None = None


class BriefingCommandResponse(BaseModel):
    """Whether a command was a briefing request, and the briefing if it was."""

    matched: bool
    intent: str
    briefing: DailyBriefing | None = None


@router.post("/briefing/command", response_model=BriefingCommandResponse)
def briefing_command(request: BriefingCommandRequest) -> BriefingCommandResponse:
    """Handle an explicit 'good morning' style command.

    Intent classification stays with the BrainEngine's rule-based resolver --
    this endpoint asks it what the user meant rather than pattern-matching on
    its own. The trigger is always an explicit command the user typed or spoke;
    there is no wake word and no always-listening microphone.
    """
    intent = brain_engine.intent_resolver.resolve(request.text)
    if intent.intent_type != "daily_briefing":
        return BriefingCommandResponse(matched=False, intent=intent.intent_type)

    return BriefingCommandResponse(
        matched=True,
        intent=intent.intent_type,
        briefing=daily_briefing_service.generate(user_name=request.user_name),
    )


@router.get("/briefing/daily", response_model=DailyBriefing)
def get_daily_briefing(user_name: str | None = None) -> DailyBriefing:
    """Generate the daily briefing from real, configured sources.

    A source that is missing, slow, or unconfigured never fails this call. It
    becomes an honest unavailable section and the rest of the briefing is still
    returned.
    """
    return daily_briefing_service.generate(user_name=user_name)


@router.post("/briefing/refresh", response_model=DailyBriefing)
def refresh_briefing(user_name: str | None = None) -> DailyBriefing:
    """Re-collect every source and produce a new briefing."""
    return daily_briefing_service.generate(user_name=user_name)


@router.get("/briefing/history", response_model=list[BriefingRecord])
def get_briefing_history() -> list[BriefingRecord]:
    """Return safe metadata for recent briefings.

    Records hold identifiers, timestamps, and availability only. No message
    bodies, audio, or secrets are ever stored or returned here.
    """
    return briefing_store.history()


@router.post("/briefing/{briefing_id}/replay", response_model=BriefingRecord)
def replay_briefing(briefing_id: str) -> BriefingRecord:
    """Record that the user replayed a briefing.

    Replay re-speaks the briefing that was already generated. It does not
    re-collect sources, so a replay cannot silently change what was said.
    """
    record = briefing_store.mark_replayed(briefing_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Briefing not found.")
    return record


@router.post("/briefing/{briefing_id}/spoken", response_model=BriefingRecord)
def mark_briefing_spoken(briefing_id: str) -> BriefingRecord:
    """Record that a briefing was spoken aloud."""
    record = briefing_store.mark_spoken(briefing_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Briefing not found.")
    return record


@router.post("/briefing/{briefing_id}/dismiss", response_model=BriefingRecord)
def dismiss_briefing(briefing_id: str) -> BriefingRecord:
    """Record that the user dismissed a briefing."""
    record = briefing_store.mark_dismissed(briefing_id)
    if record is None:
        raise HTTPException(status_code=404, detail="Briefing not found.")
    return record


@router.get("/skills", response_model=list[SkillBriefingItem])
def list_skills() -> list[SkillBriefingItem]:
    """Return every known skill: installed, in review, and recommended."""
    return skill_registry.list_skills()


@router.get("/skills/status")
def skills_status() -> dict[str, Any]:
    """Return skill subsystem status."""
    skills = skill_registry.list_skills()
    return {
        "status": "ok",
        "skill_count": len(skills),
        "catalog_configured": skill_registry.catalog_configured(),
        "recommendations_source": "reviewed_local_catalog",
        "llm_may_recommend_skills": False,
        "autonomous_installation_enabled": False,
        "research_and_installation_approvals_separate": True,
    }


@router.get("/skills/{skill_id}/gate", response_model=GateResult)
def get_skill_gate(skill_id: str) -> GateResult:
    """Return how far a skill has progressed through the learning gates."""
    try:
        return skill_registry.evaluate(skill_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/skills/{skill_id}/approve-research", response_model=SkillBriefingItem)
def approve_skill_research(
    skill_id: str,
    request: SkillDecisionRequest,
) -> SkillBriefingItem:
    """Raise a research approval for a skill.

    Research means reading public documentation. It clones nothing, installs
    nothing, and executes nothing. Approving it never implies approving
    installation -- that is a separate decision.
    """
    try:
        return skill_registry.request_approval(
            skill_id,
            SkillApprovalKind.RESEARCH,
            reason=request.reason,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/skills/{skill_id}/approve-installation", response_model=SkillBriefingItem)
def approve_skill_installation(
    skill_id: str,
    request: SkillDecisionRequest,
) -> SkillBriefingItem:
    """Raise an installation approval for a skill.

    This records the request for a decision. It does not install anything --
    Jarvis has no autonomous installation path.
    """
    try:
        return skill_registry.request_approval(
            skill_id,
            SkillApprovalKind.INSTALLATION,
            reason=request.reason,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error


@router.post("/skills/{skill_id}/reject", response_model=SkillBriefingItem)
def reject_skill(skill_id: str, request: SkillDecisionRequest) -> SkillBriefingItem:
    """Record that the user rejected a skill."""
    try:
        return skill_registry.reject(
            skill_id,
            reason=request.reason or "Rejected by the user.",
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
