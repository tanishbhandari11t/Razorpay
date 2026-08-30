from __future__ import annotations

"""
RecoverAI recovery executor — fail-closed + idempotent.

Only payment_link is eligible for the future controlled pilot.
While shadow/pilot disabled, every call BLOCKS and never hits Razorpay.
"""

from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.intervention import Intervention
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.controlled_pilot import evaluate_controlled_pilot
from app.services.execution_gate import check_execution_gate, load_execution_gate
from app.services.failure_mapping import map_razorpay_failure_reason
from app.services.kill_switch import kill_switch_armed
from app.services.recovery_agent import extract_agent_plan, run_agent_for_decision
from ml.src.razorpay_failure_taxonomy import classify_razorpay_failure


def execution_key(case_id: str, action: str) -> str:
    return f"recoverai:{case_id}:{action}"


@dataclass(frozen=True)
class ExecutionResult:
    executed: bool
    status: str
    action: str | None
    reason: str
    case_id: str
    payment_link_id: str | None = None
    provider_response: dict[str, Any] | None = None
    checks: dict[str, bool] | None = None
    idempotent_replay: bool = False
    execution_key: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "executed": bool(self.executed),
            "status": self.status,
            "action": self.action,
            "reason": self.reason,
            "case_id": self.case_id,
            "payment_link_id": self.payment_link_id,
            "provider_response": self.provider_response,
            "checks": self.checks or {},
            "execution_mode": load_execution_gate()["execution"]["mode"],
            "idempotent_replay": self.idempotent_replay,
            "execution_key": self.execution_key,
            "observed_at": datetime.now(UTC).isoformat(),
        }


def _latest_decision(session: Session, recovery_case_id: str) -> AgentDecision | None:
    return session.scalar(
        select(AgentDecision)
        .where(AgentDecision.recovery_case_id == recovery_case_id)
        .order_by(AgentDecision.created_at.desc())
    )


def _existing_execution(
    session: Session,
    *,
    payment_id: str,
    key: str,
) -> Intervention | None:
    return session.scalar(
        select(Intervention).where(
            Intervention.payment_id == payment_id,
            Intervention.reason == key,
        )
    )


def execute_approved_action(
    session: Session,
    *,
    case_id: str,
    approved_action: str | None = None,
) -> ExecutionResult:
    row = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(RecoveryCase.id == case_id)
    ).one_or_none()
    if row is None:
        return ExecutionResult(
            executed=False,
            status="blocked",
            action=approved_action,
            reason="case_not_found",
            case_id=case_id,
        )

    recovery_case, payment, customer = row
    decision = _latest_decision(session, case_id)
    if decision is None:
        return ExecutionResult(
            executed=False,
            status="blocked",
            action=approved_action,
            reason="no_shadow_decision",
            case_id=case_id,
        )

    action = approved_action or decision.selected_action
    key = execution_key(case_id, action or "none")
    existing = _existing_execution(session, payment_id=payment.id, key=key)
    if existing is not None:
        return ExecutionResult(
            executed=existing.status == "executed",
            status=existing.status,
            action=existing.type,
            reason="idempotent_replay",
            case_id=case_id,
            payment_link_id=None,
            idempotent_replay=True,
            execution_key=key,
            provider_response={"intervention_id": existing.id},
        )

    if action != decision.selected_action:
        return ExecutionResult(
            executed=False,
            status="blocked",
            action=action,
            reason="action_does_not_match_policy",
            case_id=case_id,
            execution_key=key,
        )

    if kill_switch_armed():
        session.add(
            Intervention(
                payment_id=payment.id,
                agent_decision_id=decision.id,
                type=action or "none",
                reason=key,
                attempt_number=1,
                status="blocked",
                cost=Decimal("0"),
            )
        )
        session.flush()
        return ExecutionResult(
            executed=False,
            status="blocked",
            action=action,
            reason="global_kill_switch",
            case_id=case_id,
            execution_key=key,
            checks={"kill_switch_off": False},
        )

    gate = check_execution_gate(action)
    taxonomy = classify_razorpay_failure(
        payment.failure_reason,
        fraud_flag=int((decision.features_snapshot or {}).get("fraud_flag") or 0),
    )
    feature_support_ok = True
    for check in decision.risk_checks or []:
        if isinstance(check, dict) and check.get("name") == "phase12_compatibility_gate":
            feature_support_ok = bool(check.get("passed"))
            break

    amount_inr = float(payment.amount) / 100.0
    margin = decision.decision_margin
    decision_margin_ok = margin is None or float(margin) >= 0.0

    pilot = evaluate_controlled_pilot(
        action=action,
        model_ready=False,
        taxonomy_known=taxonomy.taxonomy not in {None, "", "unknown"},
        feature_support_ok=feature_support_ok,
        fraud_pass=taxonomy.taxonomy != "fraud_risk",
        risk_pass=bool(decision.risk_checks_passed),
        decision_margin_ok=decision_margin_ok,
        amount_inr=amount_inr,
        attempt_count=0,
        cooldown_satisfied=True,
        daily_actions=0,
        customer_actions=0,
    )

    checks = {
        "kill_switch_off": not kill_switch_armed(),
        "execution_gate_allowed": gate.allowed,
        "pilot_allowed": pilot.allowed,
        "action_is_payment_link": action == "payment_link",
        "payment_still_failed": payment.status.lower() == "failed",
        "taxonomy_execution_allowed": bool(taxonomy.execution_allowed),
        "not_unknown_taxonomy": taxonomy.taxonomy != "unknown",
        "not_fraud": taxonomy.taxonomy != "fraud_risk",
        **pilot.checks,
    }

    block_reason = None
    if not gate.allowed:
        block_reason = gate.reason
    elif not pilot.allowed:
        block_reason = pilot.reason
    elif action != "payment_link":
        block_reason = "only_payment_link_allowed_in_pilot"
    elif payment.status.lower() != "failed":
        block_reason = "payment_not_failed"
    elif not taxonomy.execution_allowed or taxonomy.taxonomy == "unknown":
        block_reason = "taxonomy_blocked"
    else:
        block_reason = "provider_call_not_enabled"

    # Persist a blocked/would-execute record so retries are idempotent.
    session.add(
        Intervention(
            payment_id=payment.id,
            agent_decision_id=decision.id,
            type=action or "none",
            reason=key,
            attempt_number=1,
            status="blocked",
            cost=Decimal("0"),
        )
    )
    session.flush()
    ensure_agent_plan_before_execute(session, decision)

    return ExecutionResult(
        executed=False,
        status="blocked",
        action=action,
        reason=block_reason,
        case_id=case_id,
        checks=checks,
        execution_key=key,
        provider_response={
            "note": (
                "Payment-link creation remains behind the controlled pilot. "
                "Collect attributed evidence before enabling."
            ),
            "customer_id": customer.id,
            "amount_inr": amount_inr,
            "policy_action": action,
            "legacy_failure_class": map_razorpay_failure_reason(
                payment.failure_reason
            ),
            "agent_plan": extract_agent_plan(decision),
        },
    )


def ensure_agent_plan_before_execute(
    session: Session,
    decision: AgentDecision,
) -> dict[str, Any] | None:
    plan = extract_agent_plan(decision)
    if plan:
        return plan
    result = run_agent_for_decision(session, decision)
    session.flush()
    return result.as_dict()
