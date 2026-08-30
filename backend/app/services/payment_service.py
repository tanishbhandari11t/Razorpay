from __future__ import annotations

from datetime import UTC, datetime
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


def extract_razorpay_failure_reason(entity: dict[str, Any]) -> str:
    """
    Prefer the most specific legitimate Razorpay failure signal.

    Generic `payment_failed` alone stays generic — taxonomy will map it to
    unknown and BLOCK execution. Descriptions are only used when they contain
    a known token; we never invent semantics.
    """
    error_reason = str(entity.get("error_reason") or "").strip()
    error_code = str(entity.get("error_code") or "").strip()
    error_description = str(entity.get("error_description") or "").strip()
    candidates = [error_reason, error_code, error_description]
    for candidate in candidates:
        if not candidate:
            continue
        normalized = candidate.lower()
        if normalized not in {"payment_failed", "bad_request_error", "unknown"}:
            return candidate
    # Fall back to description only for keyword rescue of known reasons.
    blob = f"{error_reason} {error_description} {error_code}".lower()
    for token in (
        "insufficient_funds",
        "authentication_failed",
        "otp_failed",
        "card_declined",
        "do_not_honour",
        "gateway_timeout",
        "server_error",
        "suspected_fraud",
        "invalid_vpa",
    ):
        if token in blob.replace(" ", "_"):
            return token
    return error_reason or error_code or error_description or "unknown"


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
    if entity.get("created_at") is not None:
        payment.created_at = datetime.fromtimestamp(
            int(entity["created_at"]),
            tz=UTC,
        )

    if event_type in {"payment.failed", "subscription.pending"}:
        payment.status = "failed"
        payment.failure_reason = extract_razorpay_failure_reason(entity)
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
