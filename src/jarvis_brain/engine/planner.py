from jarvis_platform.schemas.plan import ExecutionPlan, PlanStep


class Planner:
    """Create simple multi-step execution plans from known goal phrases.

    Jarvis Brain v1 uses rule-based planning only. This planner does not call
    an LLM, execute tools, or request approvals. It only turns a user goal into
    a structured plan that later kernel modules can inspect.
    """

    def create_plan(self, user_goal: str) -> ExecutionPlan:
        """Create an execution plan for a user goal."""
        normalized_goal = user_goal.strip().lower()
        steps: list[PlanStep] = []

        if (
            "prepare my morning" in normalized_goal
            or "prepare me for tomorrow" in normalized_goal
        ):
            steps.append(
                self._create_step(
                    step_number=1,
                    action="list_events",
                    reason="Review upcoming calendar events.",
                )
            )
            if "finance" in normalized_goal:
                steps.append(
                    self._create_step(
                        step_number=len(steps) + 1,
                        action="summarize_spending",
                        reason="Summarize finances mentioned in the goal.",
                    )
                )
            if "files" in normalized_goal:
                steps.append(
                    self._create_step(
                        step_number=len(steps) + 1,
                        action="list_files",
                        reason="List workspace files mentioned in the goal.",
                    )
                )

        elif "plan my day" in normalized_goal:
            steps.append(
                self._create_step(
                    step_number=1,
                    action="list_events",
                    reason="Review today's calendar events.",
                )
            )
            if "email" in normalized_goal:
                steps.append(
                    self._create_step(
                        step_number=len(steps) + 1,
                        action="draft_email",
                        target="email",
                        reason="Draft an email mentioned in the goal.",
                    )
                )

        elif "secure my home" in normalized_goal:
            steps.extend(
                [
                    self._create_step(
                        step_number=1,
                        action="lock_door",
                        target="doors",
                        requires_approval=True,
                        reason="Lock the doors as part of securing the home.",
                    ),
                    self._create_step(
                        step_number=2,
                        action="turn_off_light",
                        target="lights",
                        reason="Turn off lights as part of securing the home.",
                    ),
                ]
            )

        elif "review my finances" in normalized_goal:
            steps.extend(
                [
                    self._create_step(
                        step_number=1,
                        action="list_accounts",
                        reason="List account names for a finance review.",
                    ),
                    self._create_step(
                        step_number=2,
                        action="summarize_spending",
                        reason="Summarize recent spending.",
                    ),
                    self._create_step(
                        step_number=3,
                        action="detect_subscriptions",
                        reason="Find recurring subscriptions.",
                    ),
                ]
            )

        elif "clean my workspace" in normalized_goal:
            steps.append(
                self._create_step(
                    step_number=1,
                    action="list_files",
                    reason="Inspect workspace files before cleanup.",
                )
            )

        status = "pending"
        if not steps:
            status = "needs_clarification"

        return ExecutionPlan(
            plan_id=self._create_plan_id(normalized_goal),
            user_goal=user_goal,
            steps=steps,
            status=status,
        )

    def _create_step(
        self,
        step_number: int,
        action: str,
        target: str | None = None,
        payload: dict | None = None,
        requires_approval: bool = False,
        reason: str | None = None,
    ) -> PlanStep:
        """Create a plan step with a stable v1 step id."""
        return PlanStep(
            step_id=f"step-{step_number}",
            action=action,
            target=target,
            payload=payload or {},
            requires_approval=requires_approval,
            reason=reason,
        )

    def _create_plan_id(self, normalized_goal: str) -> str:
        """Create a readable plan id from the normalized goal."""
        slug = normalized_goal.replace(" ", "-")
        if not slug:
            return "plan-empty-goal"

        return f"plan-{slug}"
