class NaturalResponseAdapter:
    """Convert internal brain outcomes into natural user-facing messages.

    This adapter is rule-based for Jarvis Brain v1. It keeps response wording in
    one place so the engine can focus on state, safety, and task coordination.
    """

    def action_success(self, action: str, target: str | None = None) -> str:
        """Return a friendly message for a successful action."""
        if action == "open_website":
            return f"Opening {self._format_target(target)} now."

        if action == "open_app":
            return f"Opening {self._format_target(target)} now."

        if action == "create_note":
            return f"I created the note: {self._format_target(target)}."

        if target is not None:
            return f"Done: {action} {target}."

        return f"Done: {action}."

    def approval_required(
        self,
        action: str,
        target: str | None = None,
        reason: str | None = None,
    ) -> str:
        """Return a message asking the user to confirm an action."""
        action_text = self._format_action(action=action, target=target)
        message = f"This action needs your confirmation before I continue: {action_text}."

        if reason is not None:
            return f"{message} Reason: {reason}"

        return message

    def action_cancelled(self, action: str | None = None, target: str | None = None) -> str:
        """Return a message for a cancelled action."""
        if action == "open_website" and target is not None:
            return f"Okay, I cancelled opening {target}."

        if action == "open_app" and target is not None:
            return f"Okay, I cancelled opening {target}."

        if action is not None:
            action_text = self._format_action(action=action, target=target)
            return f"Okay, I cancelled {action_text}."

        return "Okay, I cancelled that action."

    def nothing_waiting_for_approval(self) -> str:
        """Return a message when there is no pending approval."""
        return "There is nothing waiting for approval right now."

    def unknown_input(self) -> str:
        """Return a fallback message for input the brain cannot handle yet."""
        return "I am not sure how to handle that yet."

    def blocked_action(self, action: str, reason: str | None = None) -> str:
        """Return a safety message for a blocked action."""
        message = f"I cannot help with that action because it is blocked for safety."

        if reason is not None:
            return f"{message} Reason: {reason}"

        return message

    def error_message(self, error: str | None = None) -> str:
        """Return a friendly error message."""
        if error is not None:
            return f"Something went wrong: {error}"

        return "Something went wrong. Please try again."

    def _format_action(self, action: str, target: str | None = None) -> str:
        """Join an action and optional target for short user-facing text."""
        if target is None:
            return action

        return f"{action} {target}"

    def _format_target(self, target: str | None) -> str:
        """Return a readable target, or a generic fallback when missing."""
        if target is None:
            return "that"

        return target
