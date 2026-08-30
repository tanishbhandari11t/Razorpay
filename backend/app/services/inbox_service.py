from __future__ import annotations

"""Needs-attention inbox — merchant only sees true exceptions."""

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.promise import Promise
from app.models.recovery_case import RecoveryCase
from app.services.decline_diagnoser import diagnose_failure
from app.services.promise_service import refresh_promise_statuses


def _amount_inr(amount_minor: int) -> float:
    value = int(amount_minor or 0)
    return round(value / 100, 2) if value >= 1000 else float(value)


# Escalation classes that always surface in Needs Attention.
_BLOCK_CLASSES = {"hard_decline"}


def needs_attention(session: Session, *, limit: int = 20) -> dict[str, Any]:
    refresh_promise_statuses(session)
    items: list[dict[str, Any]] = []

    overdue = session.execute(
        select(Promise, Customer, Payment)
        .outerjoin(Customer, Promise.customer_id == Customer.id)
        .join(Payment, Promise.payment_id == Payment.id)
        .where(Promise.status == "overdue")
        .order_by(Promise.deadline.asc())
        .limit(limit)
    ).all()
    for promise, customer, payment in overdue:
        items.append(
            {
                "id": f"promise-{promise.id}",
                "kind": "promise_overdue",
                "priority": "high",
                "title": "Promise overdue",
                "customer_name": customer.name if customer else "Customer",
                "amount": _amount_inr(promise.amount),
                "reason": promise.note or "Customer promised to pay but deadline passed",
                "case_id": promise.recovery_case_id,
                "created_at": promise.deadline.isoformat() if promise.deadline else None,
            }
        )

    cases = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(Payment.status == "failed")
        .order_by(RecoveryCase.created_at.desc())
        .limit(120)
    ).all()
    decisions = session.scalars(
        select(AgentDecision).order_by(AgentDecision.created_at.desc())
    ).all()
    latest: dict[str, AgentDecision] = {}
    for decision in decisions:
        latest.setdefault(decision.recovery_case_id, decision)

    for case, payment, customer in cases:
        amount = _amount_inr(payment.amount)
        decision = latest.get(case.id)
        action = (decision.selected_action if decision else None) or ""
        diagnosis = diagnose_failure(payment.failure_reason)
        fraud = False
        if decision and isinstance(decision.features_snapshot, dict):
            fraud = bool(int(decision.features_snapshot.get("fraud_flag") or 0))

        if fraud or diagnosis["mapped_class"] == "hard_decline":
            items.append(
                {
                    "id": f"fraud-{case.id}",
                    "kind": "fraud_review",
                    "priority": "critical",
                    "title": "Blocked — do not auto-retry",
                    "customer_name": customer.name,
                    "amount": amount,
                    "reason": diagnosis["merchant_label"],
                    "case_id": case.id,
                    "created_at": case.created_at.isoformat(),
                }
            )
        elif diagnosis["recommended_action"] == "escalate_to_merchant":
            # Diagnosis is source of truth — ignore stale escalate decisions on generic fails.
            items.append(
                {
                    "id": f"escalate-{case.id}",
                    "kind": "merchant_review",
                    "priority": "medium" if amount < 10000 else "high",
                    "title": "Needs merchant review",
                    "customer_name": customer.name,
                    "amount": amount,
                    "reason": f"{diagnosis['merchant_label']} — RecoverAI paused automation",
                    "case_id": case.id,
                    "created_at": case.created_at.isoformat(),
                }
            )
        elif amount >= 15000 and diagnosis["recoverability"] == "unknown":
            items.append(
                {
                    "id": f"highvalue-{case.id}",
                    "kind": "high_value_review",
                    "priority": "low",
                    "title": "High-value customer",
                    "customer_name": customer.name,
                    "amount": amount,
                    "reason": "Optional review recommended — automation still running",
                    "case_id": case.id,
                    "created_at": case.created_at.isoformat(),
                }
            )
        if len(items) >= limit:
            break

    priority_rank = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    items.sort(key=lambda x: (priority_rank.get(x["priority"], 9), -(x["amount"] or 0)))
    items = items[:limit]
    total = round(sum(float(i["amount"]) for i in items), 2)

    open_failed = session.execute(
        select(RecoveryCase.id)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .where(Payment.status == "failed")
    ).all()
    auto_handling = max(0, len(open_failed) - len(items))

    return {
        "count": len(items),
        "total_amount": total,
        "items": items,
        "auto_handling_count": auto_handling,
        "note": (
            "Inbox is exceptions only. Generic failures get a payment link path — "
            "they do not spam Needs Attention."
        ),
    }
