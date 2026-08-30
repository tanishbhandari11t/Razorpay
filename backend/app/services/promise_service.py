from __future__ import annotations

"""Promise-to-pay lifecycle — inspired by Churnkey dunning + Novu delayed steps."""

from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.customer import Customer
from app.models.payment import Payment
from app.models.promise import Promise
from app.models.recovery_case import RecoveryCase
from app.services.communication_templates import render_template


def _amount_inr(amount_minor: int) -> float:
    value = int(amount_minor or 0)
    return round(value / 100, 2) if value >= 1000 else float(value)


def _serialize(promise: Promise, customer: Customer | None, payment: Payment | None) -> dict[str, Any]:
    return {
        "id": promise.id,
        "customer_id": promise.customer_id,
        "customer_name": customer.name if customer else "Customer",
        "payment_id": promise.payment_id,
        "recovery_case_id": promise.recovery_case_id,
        "amount": _amount_inr(promise.amount),
        "amount_minor": promise.amount,
        "note": promise.note,
        "source": promise.source,
        "language": promise.language,
        "promised_at": promise.promised_at.isoformat() if promise.promised_at else None,
        "deadline": promise.deadline.isoformat() if promise.deadline else None,
        "status": promise.status,
        "reminder_count": promise.reminder_count,
        "last_reminded_at": (
            promise.last_reminded_at.isoformat() if promise.last_reminded_at else None
        ),
        "fulfilled_at": promise.fulfilled_at.isoformat() if promise.fulfilled_at else None,
        "payment_status": payment.status if payment else None,
        "created_at": promise.created_at.isoformat() if promise.created_at else None,
    }


def create_promise(
    session: Session,
    *,
    recovery_case_id: str | None = None,
    payment_id: str | None = None,
    days: int = 1,
    note: str | None = None,
    language: str = "hinglish",
    source: str = "merchant",
) -> dict[str, Any]:
    recovery_case = None
    payment = None
    if recovery_case_id:
        recovery_case = session.get(RecoveryCase, recovery_case_id)
        if recovery_case is None:
            return {"ok": False, "reason": "case_not_found"}
        payment = session.get(Payment, recovery_case.payment_id)
    elif payment_id:
        payment = session.get(Payment, payment_id)
        recovery_case = session.scalar(
            select(RecoveryCase).where(RecoveryCase.payment_id == payment_id)
        )
    else:
        return {"ok": False, "reason": "case_or_payment_required"}

    if payment is None:
        return {"ok": False, "reason": "payment_not_found"}

    existing = session.scalar(
        select(Promise).where(
            Promise.payment_id == payment.id,
            Promise.status.in_(("pending", "reminded", "overdue")),
        )
    )
    if existing is not None:
        customer = session.get(Customer, existing.customer_id) if existing.customer_id else None
        return {
            "ok": True,
            "idempotent": True,
            "promise": _serialize(existing, customer, payment),
        }

    now = datetime.now(UTC)
    promise = Promise(
        id=str(uuid4()),
        customer_id=payment.customer_id or (recovery_case.customer_id if recovery_case else None),
        payment_id=payment.id,
        recovery_case_id=recovery_case.id if recovery_case else None,
        amount=int(payment.amount or 0),
        note=note or "Customer promised to pay",
        source=source,
        language=language,
        promised_at=now,
        deadline=now + timedelta(days=max(1, days)),
        status="pending",
        reminder_count=0,
    )
    session.add(promise)
    session.flush()
    customer = session.get(Customer, promise.customer_id) if promise.customer_id else None
    return {"ok": True, "idempotent": False, "promise": _serialize(promise, customer, payment)}


def list_promises(session: Session, *, limit: int = 50) -> list[dict[str, Any]]:
    rows = session.execute(
        select(Promise, Customer, Payment)
        .outerjoin(Customer, Promise.customer_id == Customer.id)
        .join(Payment, Promise.payment_id == Payment.id)
        .order_by(Promise.deadline.asc())
        .limit(max(1, min(limit, 200)))
    ).all()
    refresh_promise_statuses(session)
    return [_serialize(promise, customer, payment) for promise, customer, payment in rows]


def promise_summary(session: Session) -> dict[str, Any]:
    refresh_promise_statuses(session)
    promises = session.scalars(select(Promise)).all()
    pending = [p for p in promises if p.status in {"pending", "reminded"}]
    overdue = [p for p in promises if p.status == "overdue"]
    fulfilled = [p for p in promises if p.status == "fulfilled"]
    return {
        "promised_amount": round(sum(_amount_inr(p.amount) for p in pending + overdue), 2),
        "collected_amount": round(sum(_amount_inr(p.amount) for p in fulfilled), 2),
        "pending_amount": round(sum(_amount_inr(p.amount) for p in pending), 2),
        "overdue_amount": round(sum(_amount_inr(p.amount) for p in overdue), 2),
        "pending_count": len(pending),
        "overdue_count": len(overdue),
        "fulfilled_count": len(fulfilled),
        "total_count": len(promises),
    }


def refresh_promise_statuses(session: Session) -> int:
    """Mark fulfilled from captures; mark overdue past deadline."""
    now = datetime.now(UTC)
    changed = 0
    promises = session.scalars(
        select(Promise).where(Promise.status.in_(("pending", "reminded", "overdue")))
    ).all()
    for promise in promises:
        payment = session.get(Payment, promise.payment_id)
        if payment and (payment.status or "").lower() in {"captured", "paid", "recovered"}:
            promise.status = "fulfilled"
            promise.fulfilled_at = now
            changed += 1
            continue
        if promise.deadline and promise.deadline < now and promise.status != "overdue":
            promise.status = "overdue"
            changed += 1
    if changed:
        session.flush()
    return changed


def due_for_reminder(session: Session) -> list[Promise]:
    refresh_promise_statuses(session)
    now = datetime.now(UTC)
    window_start = now - timedelta(hours=12)
    promises = session.scalars(
        select(Promise).where(Promise.status.in_(("pending", "reminded", "overdue")))
    ).all()
    due: list[Promise] = []
    for promise in promises:
        if promise.deadline > now + timedelta(hours=1):
            # Not yet near due
            if promise.status != "overdue":
                continue
        if promise.last_reminded_at and promise.last_reminded_at > window_start:
            continue
        if promise.reminder_count >= 3:
            continue
        due.append(promise)
    return due


def send_promise_reminder(session: Session, promise: Promise) -> dict[str, Any]:
    """Draft reminder only — shadow-safe, never sends externally."""
    customer = session.get(Customer, promise.customer_id) if promise.customer_id else None
    payment = session.get(Payment, promise.payment_id)
    draft = render_template(
        "whatsapp_reminder",
        language=promise.language or "hinglish",
        customer_name=customer.name if customer else "there",
        amount_minor=int(promise.amount or 0),
    )
    promise.reminder_count = int(promise.reminder_count or 0) + 1
    promise.last_reminded_at = datetime.now(UTC)
    if promise.status == "pending":
        promise.status = "reminded"
    session.flush()
    return {
        "ok": True,
        "promise_id": promise.id,
        "executed": False,
        "preview_only": True,
        "message": draft["message"],
        "language": draft["language"],
        "reminder_count": promise.reminder_count,
        "customer_name": customer.name if customer else None,
        "amount": _amount_inr(promise.amount),
    }


def seed_demo_promises(session: Session) -> list[dict[str, Any]]:
    """Create a few demo promises from open at-risk cases if none exist."""
    existing = session.scalar(select(Promise).limit(1))
    if existing is not None:
        return []
    cases = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(Payment.status == "failed")
        .order_by(RecoveryCase.created_at.desc())
        .limit(4)
    ).all()
    created = []
    for idx, (case, payment, customer) in enumerate(cases[:3]):
        days = 1 if idx == 0 else (0 if idx == 2 else 2)
        # idx==2 → already overdue (deadline yesterday)
        result = create_promise(
            session,
            recovery_case_id=case.id,
            days=max(days, 1),
            note=f"{customer.name.split()[0]} said they'll pay soon",
            language="hinglish",
            source="demo_seed",
        )
        if result.get("ok") and not result.get("idempotent"):
            promise = session.get(Promise, result["promise"]["id"])
            if promise and idx == 2:
                promise.deadline = datetime.now(UTC) - timedelta(hours=6)
                promise.status = "overdue"
                promise.note = "Customer said today — still unpaid"
            created.append(result["promise"])
    session.flush()
    return created
