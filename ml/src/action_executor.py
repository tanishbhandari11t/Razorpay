from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum

from ml.src.policies.recovery_policy import RecoveryDecision


class ExecutionStatus(StrEnum):
    WOULD_EXECUTE = "would_execute"
    BLOCKED = "blocked"
    STOPPED = "stopped"


@dataclass(frozen=True)
class DryRunExecution:
    case_id: str
    payment_id: str
    action: str | None
    status: ExecutionStatus
    message: str
    executed_at: datetime
    dry_run: bool = True

    def to_dict(self) -> dict:
        value = asdict(self)
        value["status"] = self.status.value
        value["executed_at"] = self.executed_at.isoformat()
        return value


def execute_retry(decision: RecoveryDecision) -> DryRunExecution:
    return _would_execute(decision, "retry_payment")


def execute_payment_link(decision: RecoveryDecision) -> DryRunExecution:
    return _would_execute(decision, "payment_link")


def execute_whatsapp(decision: RecoveryDecision) -> DryRunExecution:
    return _would_execute(decision, "whatsapp_reminder")


def execute_escalation(decision: RecoveryDecision) -> DryRunExecution:
    return _would_execute(decision, "escalate_to_merchant")


def _would_execute(
    decision: RecoveryDecision,
    action: str,
) -> DryRunExecution:
    if not decision.dry_run:
        raise RuntimeError("Phase 8 executor refuses non-dry-run decisions")
    if decision.selected_action != action:
        raise ValueError("Executor action does not match policy decision")
    return DryRunExecution(
        case_id=decision.case_id,
        payment_id=decision.payment_id,
        action=action,
        status=ExecutionStatus.WOULD_EXECUTE,
        message=f"WOULD_EXECUTE:{action}",
        executed_at=datetime.now(UTC),
    )


def execute_decision(decision: RecoveryDecision) -> DryRunExecution:
    if not decision.dry_run:
        raise RuntimeError("Phase 8 executor is dry-run only")
    if decision.selected_action is None:
        status = (
            ExecutionStatus.STOPPED
            if decision.decision_type.value == "stop"
            else ExecutionStatus.BLOCKED
        )
        return DryRunExecution(
            case_id=decision.case_id,
            payment_id=decision.payment_id,
            action=None,
            status=status,
            message="NO_ACTION:" + ",".join(decision.reasons),
            executed_at=datetime.now(UTC),
        )
    executors = {
        "retry_payment": execute_retry,
        "payment_link": execute_payment_link,
        "whatsapp_reminder": execute_whatsapp,
        "escalate_to_merchant": execute_escalation,
    }
    return executors[decision.selected_action](decision)
