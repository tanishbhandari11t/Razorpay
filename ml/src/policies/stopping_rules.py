from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any


@dataclass(frozen=True)
class InterventionAttempt:
    action: str
    status: str
    executed_at: datetime
    cost_inr: float = 0.0


@dataclass
class RecoveryPolicyContext:
    case_id: str
    payment_id: str
    amount_inr: float
    payment_status: str
    failure_reason: str | None
    fraud_flag: int
    case_created_at: datetime
    now: datetime = field(default_factory=lambda: datetime.now(UTC))
    customer_contact_available: bool = False
    customer_opted_out: bool = False
    valid_payment_context: bool = True
    attempts: list[InterventionAttempt] = field(default_factory=list)


@dataclass(frozen=True)
class StoppingResult:
    stop: bool
    reason: str | None
    forced_action: str | None = None


def _aware(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def count_action_attempts(
    context: RecoveryPolicyContext,
    action: str,
) -> int:
    return sum(attempt.action == action for attempt in context.attempts)


def evaluate_stopping_rules(
    context: RecoveryPolicyContext,
    config: dict[str, Any],
) -> StoppingResult:
    if context.payment_status.lower() in {"captured", "recovered", "paid"}:
        return StoppingResult(True, "payment_already_recovered")
    if context.customer_opted_out:
        return StoppingResult(True, "customer_opted_out")

    limits = config["case_limits"]
    if len(context.attempts) >= int(limits["max_actions"]):
        return StoppingResult(True, "maximum_intervention_budget_reached")

    age = _aware(context.now) - _aware(context.case_created_at)
    if age > timedelta(hours=float(limits["recovery_window_hours"])):
        return StoppingResult(True, "recovery_window_expired")

    if int(context.fraud_flag) == 1:
        return StoppingResult(
            False,
            "fraud_requires_manual_escalation",
            forced_action="escalate_to_merchant",
        )

    recent_window = timedelta(
        hours=float(limits["recent_intervention_window_hours"])
    )
    recent_attempts = sum(
        _aware(context.now) - _aware(attempt.executed_at) <= recent_window
        for attempt in context.attempts
    )
    if recent_attempts >= int(limits["max_recent_interventions"]):
        if count_action_attempts(context, "escalate_to_merchant") == 0:
            return StoppingResult(
                False,
                "recent_intervention_limit_requires_escalation",
                forced_action="escalate_to_merchant",
            )
        return StoppingResult(True, "recent_intervention_limit_reached")
    return StoppingResult(False, None)


def action_control_reason(
    context: RecoveryPolicyContext,
    action: str,
    config: dict[str, Any],
) -> str | None:
    assumptions = config["interventions"][action]
    if not assumptions["enabled"]:
        return "intervention_disabled"
    attempts = [
        attempt for attempt in context.attempts if attempt.action == action
    ]
    if len(attempts) >= int(assumptions["max_attempts"]):
        return "maximum_action_attempts_reached"
    if attempts:
        last_attempt = max(_aware(value.executed_at) for value in attempts)
        cooldown_until = last_attempt + timedelta(
            hours=float(assumptions["cooldown_hours"])
        )
        if _aware(context.now) < cooldown_until:
            return f"cooldown_until:{cooldown_until.isoformat()}"
    return None
