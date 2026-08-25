from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase


def save_created_order(
    session: Session,
    order: dict[str, Any],
) -> Payment:
    payment = session.scalar(
        select(Payment).where(Payment.razorpay_order_id == str(order["id"]))
    )
    if payment is None:
        payment = Payment(
            razorpay_order_id=str(order["id"]),
            amount=int(order["amount"]),
            currency=str(order.get("currency", "INR")),
            status=str(order.get("status", "created")),
        )
        session.add(payment)
    else:
        payment.status = str(order.get("status", payment.status))
    session.flush()
    return payment


def upsert_payment_from_webhook(
    session: Session,
    entity: dict[str, Any],
    *,
    event_type: str,
    customer_id: str | None,
    fallback_payment_id: str,
) -> Payment:
    provider_payment_id = str(entity.get("id") or fallback_payment_id)
    order_id = entity.get("order_id")

    payment = session.scalar(
        select(Payment).where(Payment.razorpay_payment_id == provider_payment_id)
    )
    if payment is None and order_id:
        payment = session.scalar(
            select(Payment).where(Payment.razorpay_order_id == str(order_id))
        )
    if payment is None:
        payment = Payment(
            razorpay_payment_id=provider_payment_id,
            razorpay_order_id=str(order_id) if order_id else None,
            customer_id=customer_id,
            amount=int(entity.get("amount") or 0),
            currency=str(entity.get("currency") or "INR"),
            status="created",
        )
        session.add(payment)

    payment.razorpay_payment_id = provider_payment_id
    payment.customer_id = payment.customer_id or customer_id
    payment.razorpay_subscription_id = (
        str(entity["subscription_id"]) if entity.get("subscription_id") else None
    )
    payment.amount = int(entity.get("amount") or payment.amount)
    payment.currency = str(entity.get("currency") or payment.currency)
    payment.method = str(entity["method"]) if entity.get("method") else payment.method

    if event_type in {"payment.failed", "subscription.pending"}:
        payment.status = "failed"
        payment.failure_reason = str(
            entity.get("error_reason")
            or entity.get("error_code")
            or entity.get("error_description")
            or "unknown"
        )
    elif event_type in {
        "payment.captured",
        "subscription.charged",
        "payment_link.paid",
    }:
        payment.status = "captured"
        payment.failure_reason = None
    else:
        payment.status = str(entity.get("status") or payment.status)

    session.flush()
    return payment


def recent_payments(session: Session, limit: int = 25) -> list[dict[str, Any]]:
    payments = session.scalars(
        select(Payment).order_by(Payment.updated_at.desc()).limit(limit)
    ).all()
    return [
        {
            "id": payment.id,
            "entity_id": payment.razorpay_payment_id,
            "entity_type": "payment",
            "order_id": payment.razorpay_order_id,
            "subscription_id": payment.razorpay_subscription_id,
            "status": payment.status,
            "amount": payment.amount,
            "currency": payment.currency,
            "failure_reason": payment.failure_reason,
            "updated_at": payment.updated_at.isoformat(),
            "last_event": (
                "payment.failed"
                if payment.status == "failed"
                else "payment.captured"
                if payment.status == "captured"
                else payment.status
            ),
        }
        for payment in payments
    ]


def recent_recovery_cases(
    session: Session,
    limit: int = 25,
) -> list[dict[str, Any]]:
    rows = session.execute(
        select(RecoveryCase, Payment)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .order_by(RecoveryCase.created_at.desc())
        .limit(limit)
    ).all()
    return [
        {
            "id": recovery_case.id,
            "payment_id": payment.id,
            "razorpay_payment_id": payment.razorpay_payment_id,
            "customer_id": recovery_case.customer_id,
            "amount": payment.amount,
            "currency": payment.currency,
            "failure_reason": payment.failure_reason,
            "status": recovery_case.status,
            "source_event_id": recovery_case.source_event_id,
            "created_at": recovery_case.created_at.isoformat(),
        }
        for recovery_case, payment in rows
    ]
