from enum import Enum
from typing import List, Optional


class BrainState(Enum):
    IDLE = "IDLE"
    RECEIVED_REQUEST = "RECEIVED_REQUEST"
    UNDERSTANDING = "UNDERSTANDING"
    DECIDING = "DECIDING"
    RISK_CHECKING = "RISK_CHECKING"
    WAITING_APPROVAL = "WAITING_APPROVAL"
    EXECUTING = "EXECUTING"
    RESPONDING = "RESPONDING"
    UPDATING_MEMORY = "UPDATING_MEMORY"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"


class BrainStateMachine:
    def __init__(self) -> None:
        self.current_state = BrainState.IDLE
        self.previous_state: Optional[BrainState] = None
        self.state_history: List[BrainState] = [BrainState.IDLE]
        self._allowed_transitions = {
            BrainState.IDLE: {BrainState.RECEIVED_REQUEST, BrainState.FAILED},
            BrainState.RECEIVED_REQUEST: {BrainState.UNDERSTANDING, BrainState.FAILED},
            BrainState.UNDERSTANDING: {BrainState.DECIDING, BrainState.FAILED},
            BrainState.DECIDING: {BrainState.RISK_CHECKING, BrainState.FAILED},
            BrainState.RISK_CHECKING: {BrainState.WAITING_APPROVAL, BrainState.EXECUTING, BrainState.FAILED},
            BrainState.WAITING_APPROVAL: {BrainState.EXECUTING, BrainState.FAILED},
            BrainState.EXECUTING: {BrainState.RESPONDING, BrainState.FAILED},
            BrainState.RESPONDING: {BrainState.UPDATING_MEMORY, BrainState.FAILED},
            BrainState.UPDATING_MEMORY: {BrainState.COMPLETED, BrainState.FAILED},
            BrainState.COMPLETED: {BrainState.IDLE, BrainState.FAILED},
            BrainState.FAILED: set(),
        }

    def transition_to(self, new_state: BrainState, reason: str) -> None:
        if new_state == BrainState.FAILED:
            self.previous_state = self.current_state
            self.current_state = new_state
            self.state_history.append(new_state)
            return

        if new_state not in self._allowed_transitions.get(self.current_state, set()):
            raise ValueError(
                f"Invalid transition from {self.current_state.name} to {new_state.name}"
            )

        self.previous_state = self.current_state
        self.current_state = new_state
        self.state_history.append(new_state)
