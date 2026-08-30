from __future__ import annotations

from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.payment import Payment
from app.models.payment_feature_context import PaymentFeatureContext
from app.services.features.temporal import TransactionSnapshot


TERMINAL_SUCCESS_STATUSES = {"captured", "recovered", "paid"}
TERMINAL_FAILURE_STATUSES = {"failed"}
CONTEXT_FIELDS = (
    "transaction_type",
    "merchant_category",
    "device_type",
    "network_type",
    "sender_age_group",
    "sender_state",
    "sender_bank",
)


class OnlineFeatureUnavailable(RuntimeError):
    pass


def _normalized(value: object) -> str:
    text = str(value or "").strip()
    return text if text else "UNKNOWN"


def upsert_payment_feature_context(
    session: Session,
    payment: Payment,
    entity: dict[str, Any],
) -> PaymentFeatureContext:
    context = session.scalar(
        select(PaymentFeatureContext).where(
            PaymentFeatureContext.payment_id == payment.id
        )
    )
    notes = entity.get("notes")
    notes = notes if isinstance(notes, dict) else {}
    values = {
        "transaction_type": _normalized(notes.get("transaction_type")),
        "merchant_category": _normalized(notes.get("merchant_category")),
        "device_type": _normalized(notes.get("device_type")),
        "network_type": _normalized(notes.get("network_type")),
        "sender_age_group": _normalized(notes.get("sender_age_group")),
        "sender_state": _normalized(notes.get("sender_state")),
        "sender_bank": _normalized(entity.get("bank") or notes.get("sender_bank")),
    }
    unknown_fields = [
        name for name, value in values.items() if value == "UNKNOWN"
    ]
    fraud_flag = int(bool(notes.get("fraud_flag", 0)))
    if context is None:
        context = PaymentFeatureContext(
            payment_id=payment.id,
            **values,
            fraud_flag=fraud_flag,
            unknown_fields=unknown_fields,
        )
        session.add(context)
    else:
        for name, value in values.items():
            setattr(context, name, value)
        context.fraud_flag = fraud_flag
        context.unknown_fields = unknown_fields
    session.flush()
    return context


def payment_status_for_features(status: str) -> str | None:
    normalized = status.lower()
    if normalized in TERMINAL_SUCCESS_STATUSES:
        return "SUCCESS"
    if normalized in TERMINAL_FAILURE_STATUSES:
        return "FAILED"
    return None


def load_prior_customer_history(
    session: Session,
    current_payment: Payment,
) -> list[TransactionSnapshot]:
    if current_payment.customer_id is None:
        raise OnlineFeatureUnavailable("Payment has no persistent customer ID")
    rows = session.execute(
        select(Payment, PaymentFeatureContext)
        .outerjoin(
            PaymentFeatureContext,
            PaymentFeatureContext.payment_id == Payment.id,
        )
        .where(
            Payment.customer_id == current_payment.customer_id,
            Payment.created_at < current_payment.created_at,
        )
        .order_by(Payment.created_at.asc(), Payment.id.asc())
    ).all()
    history: list[TransactionSnapshot] = []
    for payment, context in rows:
        status = payment_status_for_features(payment.status)
        if status is None:
            continue
        if context is None:
            raise OnlineFeatureUnavailable(
                "Prior terminal payment is missing feature context"
            )
        history.append(
            TransactionSnapshot(
                transaction_id=payment.id,
                customer_id=str(payment.customer_id),
                timestamp=payment.created_at,
                status=status,
                amount_inr=float(payment.amount) / 100,
                transaction_type=context.transaction_type,
                merchant_category=context.merchant_category,
                device_type=context.device_type,
                network_type=context.network_type,
                fraud_flag=context.fraud_flag,
            )
        )
    return history


def get_payment_feature_context(
    session: Session,
    payment_id: str,
) -> PaymentFeatureContext:
    context = session.scalar(
        select(PaymentFeatureContext).where(
            PaymentFeatureContext.payment_id == payment_id
        )
    )
    if context is None:
        raise OnlineFeatureUnavailable(
            "Payment feature context has not been persisted"
        )
    return context
