from __future__ import annotations

"""Build an auditable case timeline from persisted RecoverAI records."""

from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.intervention import Intervention
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.models.webhook_event import WebhookEvent
from app.services.decline_diagnoser import diagnose_failure
from app.services.recovery_agent import extract_agent_plan


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value is not None else None


def _amount_inr(amount_minor: int) -> float:
    return round(int(amount_minor) / 100, 2)


def build_case_timeline(
    session: Session,
    recovery_case_id: str,
) -> dict[str, Any]:
    row = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(RecoveryCase.id == recovery_case_id)
    ).one_or_none()
    if row is None:
        return {"case_id": recovery_case_id, "found": False, "events": []}

    recovery_case, payment, customer = row
    events: list[dict[str, Any]] = []

    if recovery_case.source_event_id:
        webhook = session.scalar(
            select(WebhookEvent).where(
                WebhookEvent.razorpay_event_id == recovery_case.source_event_id
            )
        )
        if webhook is not None:
            events.append(
                {
                    "at": _iso(webhook.last_received_at or webhook.created_at),
                    "event": "Webhook verified",
                    "detail": f"{webhook.event_type} · signature checked",
                    "tone": "neutral",
                }
            )

    events.append(
        {
            "at": _iso(recovery_case.created_at),
            "event": "Recovery case created",
            "detail": f"Payment {payment.razorpay_payment_id or payment.id} marked {payment.status}",
            "tone": "warning" if payment.status == "failed" else "neutral",
        }
    )

    decisions = session.scalars(
        select(AgentDecision)
        .where(AgentDecision.recovery_case_id == recovery_case_id)
        .order_by(AgentDecision.created_at.asc())
    ).all()
    latest: AgentDecision | None = None
    for decision in decisions:
        latest = decision
        probs = decision.predicted_probabilities or {}
        selected = decision.selected_action
        score = probs.get(selected) if selected else None
        detail = f"Action={selected or 'none'}"
        if score is not None:
            detail = f"{detail} · P={float(score):.2f}"
        events.append(
            {
                "at": _iso(decision.created_at),
                "event": "XGBoost + policy decision",
                "detail": detail,
                "tone": "positive",
            }
        )
        events.append(
            {
                "at": _iso(decision.created_at),
                "event": "Safety gate",
                "detail": (
                    "ALLOW (shadow preview only)"
                    if decision.decision_type in {"allow", "fallback"}
                    else f"BLOCK ({decision.decision_type})"
                ),
                "tone": "positive" if decision.risk_checks_passed else "warning",
            }
        )
        plan = extract_agent_plan(decision)
        if plan and plan.get("message"):
            events.append(
                {
                    "at": _iso(decision.created_at),
                    "event": "Agent communication drafted",
                    "detail": str(plan["message"])[:180],
                    "tone": "neutral",
                }
            )
        elif plan:
            events.append(
                {
                    "at": _iso(decision.created_at),
                    "event": "Agent communication skipped",
                    "detail": plan.get("communication_status") or "no_customer_message",
                    "tone": "neutral",
                }
            )

    interventions = session.scalars(
        select(Intervention)
        .where(Intervention.payment_id == payment.id)
        .order_by(Intervention.created_at.asc())
    ).all()
    for intervention in interventions:
        events.append(
            {
                "at": _iso(intervention.created_at),
                "event": f"Intervention {intervention.type}",
                "detail": f"status={intervention.status} · executed=false unless captured later",
                "tone": "warning" if intervention.status == "blocked" else "neutral",
            }
        )

    outcomes = session.scalars(
        select(InterventionOutcome)
        .where(InterventionOutcome.recovery_case_id == recovery_case_id)
        .order_by(InterventionOutcome.created_at.asc())
    ).all()
    recovered_amount = 0.0
    outcome_label = None
    for outcome in outcomes:
        outcome_label = outcome.outcome_state
        events.append(
            {
                "at": _iso(outcome.created_at),
                "event": "Outcome tracking started",
                "detail": f"state={outcome.outcome_state} · attempted={outcome.attempted}",
                "tone": "neutral",
            }
        )
        if outcome.payment_status_after_24h:
            events.append(
                {
                    "at": _iso(outcome.last_observed_at or outcome.updated_at),
                    "event": "24h snapshot",
                    "detail": f"payment_status={outcome.payment_status_after_24h}",
                    "tone": "neutral",
                }
            )
        if outcome.payment_status_after_48h:
            events.append(
                {
                    "at": _iso(outcome.last_observed_at or outcome.updated_at),
                    "event": "48h snapshot",
                    "detail": f"payment_status={outcome.payment_status_after_48h}",
                    "tone": "neutral",
                }
            )
        if outcome.outcome_state == "attributed_intervention_recovery" or (
            outcome.payment_recovered is True and outcome.attempted
        ):
            recovered_amount = _amount_inr(outcome.recovered_amount_minor or payment.amount)
            events.append(
                {
                    "at": _iso(outcome.recovery_timestamp or outcome.outcome_at),
                    "event": "Attributed recovery confirmed",
                    "detail": f"₹{recovered_amount:,.0f} recovered",
                    "tone": "positive",
                }
            )
        elif outcome.natural_recovery_observed:
            recovered_amount = _amount_inr(outcome.recovered_amount_minor or payment.amount)
            events.append(
                {
                    "at": _iso(outcome.recovery_timestamp or outcome.outcome_at),
                    "event": "Observational recovery",
                    "detail": "Independent capture — not attributed to intervention",
                    "tone": "positive",
                }
            )

    if payment.status.lower() in {"captured", "paid", "recovered"} and recovered_amount <= 0:
        recovered_amount = _amount_inr(payment.amount)
        events.append(
            {
                "at": _iso(payment.updated_at),
                "event": "Payment captured",
                "detail": f"Provider status={payment.status}",
                "tone": "positive",
            }
        )

    events = [event for event in events if event.get("at")]
    events.sort(key=lambda item: item["at"] or "")

    plan = extract_agent_plan(latest) if latest else None
    selected = latest.selected_action if latest else None
    score = None
    if latest and selected:
        raw = (latest.predicted_probabilities or {}).get(selected)
        score = float(raw) if raw is not None else None

    diagnosis = diagnose_failure(payment.failure_reason)

    return {
        "found": True,
        "case_id": recovery_case.id,
        "customer_name": customer.name,
        "amount": _amount_inr(payment.amount),
        "payment_status": payment.status,
        "failure_reason": payment.failure_reason,
        "diagnosis": diagnosis,
        "model_version": latest.model_version if latest else None,
        "policy_version": latest.policy_version if latest else None,
        "action": selected,
        "recovery_probability": score,
        "safety_status": (
            "passed" if latest and latest.risk_checks_passed else "review"
        ),
        "executed": False,
        "execution_mode": latest.execution_mode if latest else "shadow",
        "agent_message": plan.get("message") if plan else None,
        "outcome": outcome_label,
        "amount_recovered": recovered_amount
        if payment.status.lower() in {"captured", "paid", "recovered"}
        or (outcomes and any(o.payment_recovered or o.natural_recovery_observed for o in outcomes))
        else 0.0,
        "attribution": (
            "attributed_intervention_recovery"
            if any(
                o.payment_recovered and o.attempted
                for o in outcomes
            )
            else (
                "observational_recovery"
                if any(o.natural_recovery_observed for o in outcomes)
                else None
            )
        ),
        "events": events,
    }
