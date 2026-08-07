from jarvis_platform.schemas.intent_result import IntentResult


class IntentResolver:
    """Resolve raw user input into a structured rule-based intent.

    Jarvis Brain v1 uses deterministic rules only. This resolver does not call
    OpenAI, Ollama, or any other LLM.
    """

    def __init__(self) -> None:
        """Create the v1 intent rule tables."""
        self._system_status_phrases = {
            "system status",
            "what is the system status",
            "what's the system status",
            "jarvis system status",
            "jarvis what is the system status",
            "hello jarvis what is the system status",
            "hey jarvis what is the system status",
        }
        self._daily_briefing_phrases = {
            "good morning",
            "good morning jarvis",
            "hey jarvis good morning",
            "jarvis good morning",
            "morning jarvis",
            "daily briefing",
            "morning briefing",
            "give me my briefing",
            "brief me",
            "what did i miss",
            "start my day",
        }
        self._approval_confirm_phrases = {
            "yes",
            "confirm",
            "approved",
            "ok",
            "okay",
            "go ahead",
        }
        self._approval_reject_phrases = {
            "no",
            "cancel",
            "reject",
            "stop",
        }
        self._action_phrases: dict[str, tuple[str, str | None]] = {
            "open youtube": ("open_website", "YouTube"),
            "hey jarvis youtube": ("open_website", "YouTube"),
            "youtube": ("open_website", "YouTube"),
            "open google": ("open_website", "Google"),
            "open github": ("open_website", "GitHub"),
            "open unknown website": ("open_website", "Example.com"),
            "open unknown site": ("open_website", "Example.com"),
            "open example": ("open_website", "Example.com"),
            "open example.com": ("open_website", "Example.com"),
            "turn off lights": ("turn_off_light", "living room lights"),
            "turn off living room lights": ("turn_off_light", "living room lights"),
            "turn on lights": ("turn_on_light", "lights"),
            "draft email": ("draft_email", "email"),
            "send email": ("send_email", "email"),
            "list events": ("list_events", None),
            "show calendar": ("list_events", None),
            "create event": ("create_event", None),
            "summarize spending": ("summarize_spending", None),
            "pay bill": ("execute_payment", "bill"),
            "list files": ("list_files", None),
            "read file": ("read_file", None),
            "delete file": ("delete_file", None),
            "delete a file": ("delete_file", None),
            "run command": ("run_command", None),
            "run a command": ("run_command", None),
        }
        self._goal_phrases = {
            "review my finances",
            "secure my home",
            "plan my day",
            "prepare my morning",
            "prepare me for tomorrow",
        }
        self._world_intelligence_phrases: dict[str, tuple[str, str]] = {
            "world briefing": ("get_world_briefing", "global"),
            "give me a world briefing": ("get_world_briefing", "global"),
            "jarvis give me a world briefing": ("get_world_briefing", "global"),
            "what is happening in the world": ("get_world_briefing", "global"),
            "global briefing": ("get_world_briefing", "global"),
            "global updates": ("get_world_briefing", "global"),
            "world update": ("get_world_briefing", "global"),
            "world updates": ("get_world_briefing", "global"),
            "give me world updates": ("get_world_briefing", "global"),
            "give me global updates": ("get_world_briefing", "global"),
            "show what's happening in the world today": ("get_world_briefing", "global"),
            "show what is happening in the world today": ("get_world_briefing", "global"),
            "what's happening in the world today": ("get_world_briefing", "global"),
            "what is happening in the world today": ("get_world_briefing", "global"),
            "what's going on in the world today": ("get_world_briefing", "global"),
            "what is going on in the world today": ("get_world_briefing", "global"),
            "show me what's going on in the world": ("get_world_briefing", "global"),
            "show me what is going on in the world": ("get_world_briefing", "global"),
            "what happened in the world today": ("get_world_briefing", "global"),
            "show world news today": ("get_world_briefing", "global"),
            "today's world briefing": ("get_world_briefing", "global"),
            "daily world briefing": ("get_world_briefing", "global"),
            "world update today": ("get_world_briefing", "global"),
            "any cyber alerts today": ("get_cyber_alerts", "cybersecurity"),
            "show cyber alerts": ("get_cyber_alerts", "cybersecurity"),
            "security alerts": ("get_cyber_alerts", "cybersecurity"),
            "any cybersecurity updates": ("get_cyber_alerts", "cybersecurity"),
            "cybersecurity updates": ("get_cyber_alerts", "cybersecurity"),
            "cloud security alerts": ("get_cyber_alerts", "cybersecurity"),
            "what global updates matter to my project": (
                "get_project_relevant_updates",
                "Jarvis project",
            ),
            "what updates matter to jarvis": (
                "get_project_relevant_updates",
                "Jarvis project",
            ),
            "anything important for my jarvis project": (
                "get_project_relevant_updates",
                "Jarvis project",
            ),
            "what should i know today": (
                "get_project_relevant_updates",
                "Jarvis project",
            ),
            "any ai research updates": ("get_ai_research_updates", "ai_research"),
            "ai agent updates": ("get_ai_research_updates", "ai_research"),
            "new ai framework updates": ("get_ai_research_updates", "ai_research"),
            "ai research briefing": ("get_ai_research_updates", "ai_research"),
            "show me world alerts": ("get_world_alerts", "alerts"),
            "any important alerts": ("get_world_alerts", "alerts"),
            "urgent world alerts": ("get_world_alerts", "alerts"),
            "world alerts": ("get_world_alerts", "alerts"),
        }

    def resolve(self, raw_input: str) -> IntentResult:
        """Resolve raw user input into an IntentResult."""
        normalized_input = raw_input.strip().lower()
        punctuation_stripped = self._strip_punctuation(normalized_input)

        if punctuation_stripped in self._system_status_phrases:
            return self._build_result(
                raw_input=raw_input,
                intent_type="system",
                confidence=0.99,
                action="get_system_status",
                metadata={"action": "get_system_status", "target": "jarvis"},
            )

        # "Hey Jarvis, good morning." is the primary briefing trigger, so it is
        # matched on a punctuation-stripped form. This is an explicit spoken or
        # typed command -- there is no wake word and nothing is always listening.
        if punctuation_stripped in self._daily_briefing_phrases:
            return self._build_result(
                raw_input=raw_input,
                intent_type="daily_briefing",
                confidence=0.95,
                action="daily_briefing",
                metadata={"action": "daily_briefing", "target": None},
            )

        if normalized_input in self._approval_confirm_phrases:
            return self._build_result(
                raw_input=raw_input,
                intent_type="approval_confirm",
                confidence=0.95,
            )

        if normalized_input in self._approval_reject_phrases:
            return self._build_result(
                raw_input=raw_input,
                intent_type="approval_reject",
                confidence=0.95,
            )

        if normalized_input in self._action_phrases:
            action, target = self._action_phrases[normalized_input]
            return self._build_action_result(
                raw_input=raw_input,
                action=action,
                target=target,
                confidence=0.95,
            )

        if normalized_input in self._goal_phrases:
            return self._build_goal_result(
                raw_input=raw_input,
                goal=normalized_input,
                confidence=0.95,
            )

        if normalized_input in self._world_intelligence_phrases:
            action, target = self._world_intelligence_phrases[normalized_input]
            return self._build_world_intelligence_result(
                raw_input=raw_input,
                action=action,
                target=target,
                confidence=0.95,
            )

        partial_action = self._resolve_partial_action(normalized_input)
        if partial_action is not None:
            action, target = partial_action
            return self._build_action_result(
                raw_input=raw_input,
                action=action,
                target=target,
                confidence=0.75,
            )

        partial_goal = self._resolve_partial_goal(normalized_input)
        if partial_goal is not None:
            return self._build_goal_result(
                raw_input=raw_input,
                goal=partial_goal,
                confidence=0.75,
            )

        return self._build_result(
            raw_input=raw_input,
            intent_type="unknown",
            confidence=0.0,
            needs_clarification=True,
        )

    def _resolve_partial_action(self, normalized_input: str) -> tuple[str, str | None] | None:
        """Return an action for simple partial keyword matches."""
        if "youtube" in normalized_input:
            return "open_website", "YouTube"
        if "google" in normalized_input and "open" in normalized_input:
            return "open_website", "Google"
        if "github" in normalized_input and "open" in normalized_input:
            return "open_website", "GitHub"
        if "example" in normalized_input and "open" in normalized_input:
            return "open_website", "Example.com"
        if "turn off" in normalized_input and "light" in normalized_input:
            return "turn_off_light", "living room lights"
        if "turn on" in normalized_input and "light" in normalized_input:
            return "turn_on_light", "lights"
        if "draft" in normalized_input and "email" in normalized_input:
            return "draft_email", "email"
        if "send" in normalized_input and "email" in normalized_input:
            return "send_email", "email"
        if "calendar" in normalized_input or "list events" in normalized_input:
            return "list_events", None
        if "create" in normalized_input and "event" in normalized_input:
            return "create_event", None
        if "summarize" in normalized_input and "spending" in normalized_input:
            return "summarize_spending", None
        if "pay" in normalized_input and "bill" in normalized_input:
            return "execute_payment", "bill"
        if "list" in normalized_input and "file" in normalized_input:
            return "list_files", None
        if "read" in normalized_input and "file" in normalized_input:
            return "read_file", None
        if "delete" in normalized_input and "file" in normalized_input:
            return "delete_file", None

        return None

    def _resolve_partial_goal(self, normalized_input: str) -> str | None:
        """Return a goal for simple partial goal matches."""
        for goal in self._goal_phrases:
            if goal in normalized_input:
                return goal

        return None

    def _strip_punctuation(self, value: str) -> str:
        """Strip surrounding punctuation so 'Hey Jarvis, good morning.' matches."""
        return " ".join(
            word.strip(",.!?;:'\"")
            for word in value.split()
            if word.strip(",.!?;:'\"")
        )

    def _build_action_result(
        self,
        raw_input: str,
        action: str,
        target: str | None,
        confidence: float,
    ) -> IntentResult:
        """Build an action IntentResult."""
        return self._build_result(
            raw_input=raw_input,
            intent_type="action",
            confidence=confidence,
            action=action,
            target=target,
            metadata={"action": action, "target": target},
        )

    def _build_goal_result(
        self,
        raw_input: str,
        goal: str,
        confidence: float,
    ) -> IntentResult:
        """Build a goal IntentResult that should be planned."""
        return self._build_result(
            raw_input=raw_input,
            intent_type="goal",
            confidence=confidence,
            goal=goal,
            requires_plan=True,
            metadata={"goal": goal},
        )

    def _build_world_intelligence_result(
        self,
        raw_input: str,
        action: str,
        target: str,
        confidence: float,
    ) -> IntentResult:
        """Build a world-intelligence IntentResult."""
        return self._build_result(
            raw_input=raw_input,
            intent_type="world_intelligence",
            confidence=confidence,
            action=action,
            target=target,
            metadata={"action": action, "target": target},
        )

    def _build_result(
        self,
        raw_input: str,
        intent_type: str,
        confidence: float,
        action: str | None = None,
        target: str | None = None,
        goal: str | None = None,
        metadata: dict | None = None,
        requires_plan: bool = False,
        needs_clarification: bool = False,
    ) -> IntentResult:
        """Build a structured IntentResult."""
        return IntentResult(
            name=intent_type,
            intent_type=intent_type,
            action=action,
            target=target,
            goal=goal,
            raw_input=raw_input,
            confidence=confidence,
            metadata=metadata or {},
            requires_plan=requires_plan,
            needs_clarification=needs_clarification,
        )
