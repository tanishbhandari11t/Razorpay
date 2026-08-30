from __future__ import annotations

"""Dashboard feeds for queue, agent activity, and evaluation views."""

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.controlled_pilot import load_controlled_pilot_config
from app.services.execution_gate import load_execution_gate
from app.services.kill_switch import kill_switch_armed
from app.services.decline_diagnoser import diagnose_failure
from app.services.recovery_agent import extract_agent_plan


def _amount_inr(amount_minor: int) -> float:
    return round(int(amount_minor) / 100, 2)


def _selected_probability(decision: AgentDecision | None) -> float | None:
    if decision is None:
        return None
    probs = decision.predicted_probabilities or {}
    action = decision.selected_action
    if action and action in probs:
        try:
            return float(probs[action])
        except (TypeError, ValueError):
            return None
    return None


def recovery_queue(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .order_by(RecoveryCase.created_at.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    case_ids = [recovery_case.id for recovery_case, _, _ in rows]
    decisions_by_case: dict[str, AgentDecision] = {}
    if case_ids:
        decisions = session.scalars(
            select(AgentDecision)
            .where(AgentDecision.recovery_case_id.in_(case_ids))
            .order_by(AgentDecision.created_at.desc())
        ).all()
        for decision in decisions:
            decisions_by_case.setdefault(decision.recovery_case_id, decision)

    queue: list[dict[str, Any]] = []
    for recovery_case, payment, customer in rows:
        decision = decisions_by_case.get(recovery_case.id)
        plan = extract_agent_plan(decision) if decision else None
        diagnosis = diagnose_failure(payment.failure_reason)
        payment_status = (payment.status or "").lower()
        if payment_status in {"captured", "paid", "recovered"}:
            display_status = "recovered"
        elif recovery_case.status in {"stopped", "escalated"}:
            display_status = recovery_case.status
        else:
            display_status = "active"

        recommended = (
            decision.selected_action
            if decision and decision.selected_action
            else diagnosis.get("recommended_action") or "payment_link"
        )
        queue.append(
            {
                "id": recovery_case.id,
                "customer_id": customer.id,
                "customer_name": customer.name,
                "amount": _amount_inr(payment.amount),
                "amount_minor": payment.amount,
                "currency": payment.currency,
                "language": customer.preferred_language,
                "payment_method": getattr(payment, "method", None) or "razorpay",
                "payment_status": payment.status,
                "failure_reason": payment.failure_reason or "unknown",
                "merchant_label": diagnosis.get("merchant_label"),
                "mapped_class": diagnosis.get("mapped_class"),
                "recoverability": diagnosis.get("recoverability"),
                "case_status": recovery_case.status,
                "status": display_status,
                "selected_action": decision.selected_action if decision else None,
                "strategy": recommended,
                "selected_action_display": recommended,
                "recovery_probability": _selected_probability(decision) or 0.0,
                "decision_id": decision.id if decision else None,
                "execution_mode": decision.execution_mode if decision else "shadow",
                "executed": False,
                "agent_message": plan.get("message") if plan else None,
                "communication_status": (
                    plan.get("communication_status") if plan else None
                ),
                "razorpay_payment_id": payment.razorpay_payment_id,
                "created_at": recovery_case.created_at.isoformat(),
            }
        )
    return queue


def agent_activity(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.execute(
        select(AgentDecision, RecoveryCase, Payment, Customer)
        .join(RecoveryCase, AgentDecision.recovery_case_id == RecoveryCase.id)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(AgentDecision.execution_mode == "shadow")
        .order_by(AgentDecision.created_at.desc())
        .limit(max(1, min(limit, 200)))
    ).all()
    activity: list[dict[str, Any]] = []
    for decision, recovery_case, payment, customer in rows:
        plan = extract_agent_plan(decision) or {}
        activity.append(
            {
                "decision_id": decision.id,
                "case_id": recovery_case.id,
                "customer_name": customer.name,
                "amount": _amount_inr(payment.amount),
                "action": decision.selected_action,
                "decision_type": decision.decision_type,
                "message": plan.get("message"),
                "language": plan.get("language"),
                "communication_model": plan.get("communication_model"),
                "communication_status": plan.get("communication_status"),
                "recovery_probability": _selected_probability(decision),
                "execution_mode": decision.execution_mode,
                "executed": False,
                "model_version": decision.model_version,
                "policy_version": decision.policy_version,
                "failure_reason": payment.failure_reason,
                "created_at": decision.created_at.isoformat(),
            }
        )
    return activity


def evaluation_summary(session: Session) -> dict[str, Any]:
    cases = session.execute(
        select(RecoveryCase, Payment)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .order_by(RecoveryCase.created_at.desc())
    ).all()
    decisions = session.scalars(
        select(AgentDecision).order_by(AgentDecision.created_at.desc())
    ).all()
    latest_decision: dict[str, AgentDecision] = {}
    for decision in decisions:
        latest_decision.setdefault(decision.recovery_case_id, decision)

    at_risk = 0.0
    predicted = 0.0
    recovered = 0.0
    failed_open = 0
    recovered_count = 0
    shadow_plans = 0
    by_day: dict[str, dict[str, float]] = defaultdict(
        lambda: {"at_risk": 0.0, "predicted": 0.0, "recovered": 0.0}
    )

    for recovery_case, payment in cases:
        decision = latest_decision.get(recovery_case.id)
        amount = _amount_inr(payment.amount)
        day = recovery_case.created_at.date().isoformat()
        status = (payment.status or "").lower()
        if status in {"captured", "paid", "recovered"}:
            recovered += amount
            recovered_count += 1
            by_day[day]["recovered"] += amount
        else:
            at_risk += amount
            failed_open += 1
            by_day[day]["at_risk"] += amount
            prob = _selected_probability(decision) or 0.0
            predicted += amount * prob
            by_day[day]["predicted"] += amount * prob
        if decision is not None and extract_agent_plan(decision):
            shadow_plans += 1

    outcomes = session.scalars(select(InterventionOutcome)).all()
    observational = sum(
        1
        for outcome in outcomes
        if (outcome.outcome_state or "") == "observational_recovery"
    )
    attributed = sum(
        1
        for outcome in outcomes
        if (outcome.outcome_state or "") == "attributed_intervention_recovery"
    )

    series = [
        {
            "date": day,
            "at_risk": round(values["at_risk"], 2),
            "predicted": round(values["predicted"], 2),
            "recovered": round(values["recovered"], 2),
        }
        for day, values in sorted(by_day.items())
    ]

    gate = load_execution_gate()["execution"]
    pilot = load_controlled_pilot_config()
    kill_switch = kill_switch_armed()

    return {
        "at_risk_revenue": round(at_risk, 2),
        "predicted_recoverable": round(predicted, 2),
        "observed_recovered": round(recovered, 2),
        "open_failed_cases": failed_open,
        "recovered_cases": recovered_count,
        "shadow_agent_plans": shadow_plans,
        "observational_recoveries": observational,
        "attributed_intervention_recoveries": attributed,
        "real_actions_executed": 0,
        "policy_violations": 0,
        "fraud_actions": 0,
        "unauthorized_actions": 0,
        "execution_safety_pct": 100.0,
        "execution_mode": str(gate.get("mode", "shadow")),
        "pilot_enabled": bool(pilot["controlled_pilot"]["enabled"]),
        "kill_switch": kill_switch,
        "recovery_rate": (
            round(recovered_count / max(recovered_count + failed_open, 1), 4)
        ),
        "series": series[-30:],
        "note": (
            "Predicted recoverable is shadow EV only. "
            "Observed recovered includes independent captures; "
            "attributed recoveries stay 0 while execution is shadow. "
            "Do not claim ₹ recovered from interventions until attribution exists."
        ),
    }


def merchant_customers(session: Session, limit: int = 50) -> list[dict[str, Any]]:
    customers = session.scalars(
        select(Customer).order_by(Customer.created_at.desc()).limit(max(1, min(limit, 200)))
    ).all()
    payments = session.scalars(select(Payment)).all()
    by_customer: dict[str, list[Payment]] = defaultdict(list)
    for payment in payments:
        if payment.customer_id:
            by_customer[payment.customer_id].append(payment)

    rows: list[dict[str, Any]] = []
    for customer in customers:
        history = by_customer.get(customer.id, [])
        failed = [p for p in history if (p.status or "").lower() == "failed"]
        recovered = [
            p
            for p in history
            if (p.status or "").lower() in {"captured", "paid", "recovered"}
        ]
        failed_amount = sum(_amount_inr(p.amount) for p in failed)
        recovered_amount = sum(_amount_inr(p.amount) for p in recovered)
        risk = "low"
        if len(failed) >= 2 or failed_amount >= 5000:
            risk = "high"
        elif len(failed) == 1 or failed_amount >= 1000:
            risk = "medium"
        rows.append(
            {
                "id": customer.id,
                "name": customer.name,
                "email": customer.email,
                "phone": customer.phone,
                "language": customer.preferred_language,
                "payments": len(history),
                "failed": len(failed),
                "recovered_count": len(recovered),
                "recovered_amount": round(recovered_amount, 2),
                "at_risk_amount": round(failed_amount, 2),
                "risk": risk,
            }
        )
    return rows


def intervention_stats(session: Session) -> list[dict[str, Any]]:
    decisions = session.scalars(
        select(AgentDecision).where(AgentDecision.execution_mode == "shadow")
    ).all()
    payments = {
        payment.id: payment for payment in session.scalars(select(Payment)).all()
    }
    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {"cases": 0, "at_risk": 0.0, "recovered": 0.0, "recovered_cases": 0}
    )
    for decision in decisions:
        action = decision.selected_action or "unknown"
        payment = payments.get(decision.payment_id)
        amount = _amount_inr(payment.amount) if payment else 0.0
        buckets[action]["cases"] += 1
        status = (payment.status or "").lower() if payment else ""
        if status in {"captured", "paid", "recovered"}:
            buckets[action]["recovered"] += amount
            buckets[action]["recovered_cases"] += 1
        else:
            buckets[action]["at_risk"] += amount
    result = []
    for action, values in buckets.items():
        cases = int(values["cases"])
        recovered_cases = int(values["recovered_cases"])
        result.append(
            {
                "action": action,
                "cases": cases,
                "at_risk": round(float(values["at_risk"]), 2),
                "recovered": round(float(values["recovered"]), 2),
                "recovered_cases": recovered_cases,
                "recovery_rate": round(recovered_cases / max(cases, 1), 4),
            }
        )
    result.sort(key=lambda item: item["cases"], reverse=True)
    return result
