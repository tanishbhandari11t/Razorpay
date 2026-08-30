from __future__ import annotations

from decimal import Decimal
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.intervention import Intervention
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.outcome_state_machine import create_outcome_for_decision


ALLOWED_ACTIONS = {
    "retry_payment",
    "payment_link",
    "whatsapp_reminder",
    "escalate_to_merchant",
}
DECISION_TYPES = {"allow", "block", "fallback", "stop"}
MAX_ACTIONS_PER_CASE = 3
DRY_RUN_COSTS = {
    "retry_payment": Decimal("1.00"),
    "payment_link": Decimal("10.00"),
    "whatsapp_reminder": Decimal("0.75"),
    "escalate_to_merchant": Decimal("25.00"),
}


class RecoveryEngineError(ValueError):
    pass


def load_recovery_case_context(
    session: Session,
    recovery_case_id: str,
) -> tuple[RecoveryCase, Payment, list[Intervention]]:
    row = session.execute(
        select(RecoveryCase, Payment)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .where(RecoveryCase.id == recovery_case_id)
    ).one_or_none()
    if row is None:
        raise RecoveryEngineError("Recovery case not found")
    recovery_case, payment = row
    attempts = session.scalars(
        select(Intervention)
        .where(Intervention.payment_id == payment.id)
        .order_by(Intervention.created_at)
    ).all()
    return recovery_case, payment, list(attempts)


def persist_dry_run_policy_decision(
    session: Session,
    *,
    recovery_case_id: str,
    decision_key: str,
    model_version: str,
    policy_version: str,
    policy_manifest_sha256: str,
    decision_type: str,
    selected_action: str | None,
    candidate_actions: dict[str, Any],
    predicted_probabilities: dict[str, float],
    expected_values: dict[str, float],
    decision_reasons: list[str],
    fallback_used: bool,
    risk_checks: dict[str, Any],
    dry_run: bool,
) -> tuple[AgentDecision, bool]:
    if not dry_run:
        raise RecoveryEngineError("Phase 8 backend accepts dry-run decisions only")
    existing = session.scalar(
        select(AgentDecision).where(
            AgentDecision.decision_key == decision_key
        )
    )
    if existing is not None:
        return existing, True
    if decision_type not in DECISION_TYPES:
        raise RecoveryEngineError("Invalid policy decision type")
    if selected_action is not None and selected_action not in ALLOWED_ACTIONS:
        raise RecoveryEngineError("Invalid selected intervention")
    if len(policy_manifest_sha256) != 64:
        raise RecoveryEngineError("Policy manifest SHA-256 is required")
    if not all(
        0 <= float(probability) <= 1
        for probability in predicted_probabilities.values()
    ):
        raise RecoveryEngineError("Predicted probabilities must be in [0, 1]")
    if selected_action is not None:
        if selected_action not in predicted_probabilities:
            raise RecoveryEngineError("Selected action is missing a prediction")
        if selected_action not in candidate_actions:
            raise RecoveryEngineError("Selected action is missing candidate audit")
        if not bool(risk_checks.get("passed")):
            raise RecoveryEngineError("Selected action failed policy risk checks")
    elif decision_type not in {"block", "stop"}:
        raise RecoveryEngineError("Actionless decision must block or stop")

    recovery_case, payment, attempts = load_recovery_case_context(
        session,
        recovery_case_id,
    )
    if payment.status == "captured" or recovery_case.status == "recovered":
        raise RecoveryEngineError("Recovered payment cannot receive an action")
    if selected_action is not None and len(attempts) >= MAX_ACTIONS_PER_CASE:
        raise RecoveryEngineError("Recovery case action budget exhausted")

    decision = AgentDecision(
        recovery_case_id=recovery_case.id,
        payment_id=payment.id,
        decision_key=decision_key,
        model_version=model_version,
        policy_version=policy_version,
        features_version="offline_phase8_features",
        policy_manifest_sha256=policy_manifest_sha256,
        execution_mode="dry_run",
        inference_status="completed",
        decision_type=decision_type,
        selected_action=selected_action,
        candidate_actions=candidate_actions,
        predicted_probabilities=predicted_probabilities,
        expected_values=expected_values,
        features_snapshot={},
        decision_reasons=decision_reasons,
        decision_margin=None,
        failure_class=None,
        fallback_used=fallback_used,
        risk_checks=risk_checks,
        risk_checks_passed=bool(risk_checks.get("passed")),
        dry_run=True,
    )
    session.add(decision)
    session.flush()
    outcome, _ = create_outcome_for_decision(session, decision)

    if selected_action is not None:
        attempt_number = int(
            session.scalar(
                select(func.count(Intervention.id)).where(
                    Intervention.payment_id == payment.id,
                    Intervention.type == selected_action,
                )
            )
            or 0
        ) + 1
        intervention = Intervention(
            payment_id=payment.id,
            agent_decision_id=decision.id,
            type=selected_action,
            reason=";".join(decision_reasons)[:500],
            attempt_number=attempt_number,
            status="would_execute",
            cost=DRY_RUN_COSTS[selected_action],
        )
        session.add(intervention)
        session.flush()
        outcome.intervention_id = intervention.id
        recovery_case.status = (
            "escalated_dry_run"
            if selected_action == "escalate_to_merchant"
            else "action_selected_dry_run"
        )
    else:
        recovery_case.status = (
            "stopped" if decision_type == "stop" else "blocked"
        )
    session.flush()
    return decision, False


def recent_agent_decisions(
    session: Session,
    recovery_case_id: str,
) -> list[dict[str, Any]]:
    decisions = session.scalars(
        select(AgentDecision)
        .where(AgentDecision.recovery_case_id == recovery_case_id)
        .order_by(AgentDecision.created_at.desc())
    ).all()
    return [
        {
            "id": decision.id,
            "decision_key": decision.decision_key,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "features_version": decision.features_version,
            "execution_mode": decision.execution_mode,
            "decision_type": decision.decision_type,
            "selected_action": decision.selected_action,
            "decision_reasons": decision.decision_reasons,
            "fallback_used": decision.fallback_used,
            "risk_checks": decision.risk_checks,
            "dry_run": decision.dry_run,
            "created_at": decision.created_at.isoformat(),
        }
        for decision in decisions
    ]
