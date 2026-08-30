from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum


class RecoveryState(StrEnum):
    AT_RISK = "at_risk"
    DIAGNOSED = "diagnosed"
    ACTION_SELECTED = "action_selected"
    ACTION_EXECUTED = "action_executed"
    RECOVERED = "recovered"
    FAILED = "failed"
    NEXT_ACTION = "next_action"
    STOPPED = "stopped"
    ESCALATED = "escalated"


ALLOWED_TRANSITIONS = {
    RecoveryState.AT_RISK: {
        RecoveryState.DIAGNOSED,
        RecoveryState.STOPPED,
    },
    RecoveryState.DIAGNOSED: {
        RecoveryState.ACTION_SELECTED,
        RecoveryState.STOPPED,
        RecoveryState.ESCALATED,
    },
    RecoveryState.ACTION_SELECTED: {
        RecoveryState.ACTION_EXECUTED,
        RecoveryState.STOPPED,
        RecoveryState.ESCALATED,
    },
    RecoveryState.ACTION_EXECUTED: {
        RecoveryState.RECOVERED,
        RecoveryState.FAILED,
        RecoveryState.ESCALATED,
    },
    RecoveryState.FAILED: {
        RecoveryState.NEXT_ACTION,
        RecoveryState.STOPPED,
        RecoveryState.ESCALATED,
    },
    RecoveryState.NEXT_ACTION: {
        RecoveryState.DIAGNOSED,
        RecoveryState.STOPPED,
        RecoveryState.ESCALATED,
    },
    RecoveryState.RECOVERED: set(),
    RecoveryState.STOPPED: set(),
    RecoveryState.ESCALATED: set(),
}


@dataclass
class RecoveryStateMachine:
    state: RecoveryState = RecoveryState.AT_RISK
    history: list[RecoveryState] = field(
        default_factory=lambda: [RecoveryState.AT_RISK]
    )

    def transition(self, target: RecoveryState) -> RecoveryState:
        if target not in ALLOWED_TRANSITIONS[self.state]:
            raise ValueError(
                f"Invalid recovery transition: {self.state} -> {target}"
            )
        self.state = target
        self.history.append(target)
        return self.state
