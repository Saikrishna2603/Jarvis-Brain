from jarvis_brain.engine.planner import Planner
from jarvis_platform.schemas.plan import ExecutionPlan


def test_planner_can_be_created() -> None:
    planner = Planner()

    assert planner is not None


def test_review_my_finances_creates_finance_steps() -> None:
    planner = Planner()

    plan = planner.create_plan("review my finances")

    actions = [step.action for step in plan.steps]
    assert actions == ["list_accounts", "summarize_spending", "detect_subscriptions"]
    assert plan.status == "pending"


def test_secure_my_home_creates_smart_home_steps() -> None:
    planner = Planner()

    plan = planner.create_plan("secure my home")

    assert [step.action for step in plan.steps] == ["lock_door", "turn_off_light"]
    assert plan.steps[0].target == "doors"
    assert plan.steps[0].requires_approval is True
    assert plan.steps[1].target == "lights"


def test_plan_my_day_creates_calendar_step() -> None:
    planner = Planner()

    plan = planner.create_plan("plan my day")

    assert len(plan.steps) == 1
    assert plan.steps[0].action == "list_events"
    assert plan.steps[0].reason == "Review today's calendar events."


def test_plan_my_day_with_email_adds_draft_email_step() -> None:
    planner = Planner()

    plan = planner.create_plan("plan my day and email my team")

    assert [step.action for step in plan.steps] == ["list_events", "draft_email"]
    assert plan.steps[1].target == "email"


def test_prepare_goal_can_add_optional_finance_and_file_steps() -> None:
    planner = Planner()

    plan = planner.create_plan("prepare my morning with finance and files")

    assert [step.action for step in plan.steps] == [
        "list_events",
        "summarize_spending",
        "list_files",
    ]


def test_unknown_goal_creates_needs_clarification_plan() -> None:
    planner = Planner()

    plan = planner.create_plan("build me a moon base")

    assert isinstance(plan, ExecutionPlan)
    assert plan.steps == []
    assert plan.status == "needs_clarification"


def test_every_step_has_a_step_id() -> None:
    planner = Planner()

    plan = planner.create_plan("review my finances")

    assert all(step.step_id for step in plan.steps)
    assert [step.step_id for step in plan.steps] == ["step-1", "step-2", "step-3"]


def test_plan_serializes_correctly() -> None:
    planner = Planner()

    plan = planner.create_plan("secure my home")
    dumped = plan.model_dump()
    json_text = plan.model_dump_json()

    assert dumped["plan_id"] == "plan-secure-my-home"
    assert dumped["steps"][0]["action"] == "lock_door"
    assert isinstance(json_text, str)
