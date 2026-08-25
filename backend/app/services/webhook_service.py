from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.database.connection import get_session
from app.models.recovery_case import RecoveryCase
from app.models.webhook_event import WebhookEvent
from app.services.customer_service import find_or_create_customer
from app.services.payment_service import upsert_payment_from_webhook


RECOVERY_EVENTS = {"payment.failed", "subscription.pending"}
RECOVERED_EVENTS = {
    "payment.captured",
    "subscription.charged",
    "payment_link.paid",
}


def _entity(payload: dict[str, Any], name: str) -> dict[str, Any]:
    wrapper = payload.get("payload", {}).get(name, {})
    entity = wrapper.get("entity", {}) if isinstance(wrapper, dict) else {}
    return entity if isinstance(entity, dict) else {}


def _payment_entity(
    payload: dict[str, Any],
    event_type: str,
    event_id: str,
) -> dict[str, Any]:
    payment = _entity(payload, "payment")
    if payment:
        return payment
    if event_type == "subscription.pending":
        subscription = _entity(payload, "subscription")
        if subscription:
            return {
                "id": f"pending:{event_id}",
                "subscription_id": subscription.get("id"),
                "amount": subscription.get("amount", 0),
                "currency": subscription.get("currency", "INR"),
                "status": "failed",
                "error_reason": "subscription_charge_failed",
            }
    return {}


def process_webhook(
    *,
    event_id: str,
    payload: dict[str, Any],
) -> dict[str, Any]:
    event_type = str(payload.get("event", "unknown"))

    # Claim the event ID first. The database unique constraint is the final
    # defense against concurrent duplicate deliveries.
    with get_session() as session:
        event = WebhookEvent(
            razorpay_event_id=event_id,
            event_type=event_type,
            payload=payload,
            processed=False,
        )
        session.add(event)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(WebhookEvent).where(
                    WebhookEvent.razorpay_event_id == event_id
                )
            )
            return {
                "event_id": event_id,
                "event_type": existing.event_type if existing else event_type,
                "duplicate": True,
                "processed": bool(existing and existing.processed),
                "recovery_case_id": None,
            }

    recovery_case_id: str | None = None
    with get_session() as session:
        event = session.scalar(
            select(WebhookEvent).where(WebhookEvent.razorpay_event_id == event_id)
        )
        if event is None:
            raise RuntimeError("Claimed webhook event could not be reloaded.")

        try:
            payment_entity = _payment_entity(payload, event_type, event_id)
            if payment_entity:
                customer = find_or_create_customer(session, payment_entity)
                payment = upsert_payment_from_webhook(
                    session,
                    payment_entity,
                    event_type=event_type,
                    customer_id=customer.id if customer else None,
                    fallback_payment_id=f"event:{event_id}",
                )

                recovery_case = session.scalar(
                    select(RecoveryCase).where(
                        RecoveryCase.payment_id == payment.id
                    )
                )
                if event_type in RECOVERY_EVENTS and recovery_case is None:
                    recovery_case = RecoveryCase(
                        payment_id=payment.id,
                        customer_id=payment.customer_id,
                        status="at_risk",
                        source_event_id=event_id,
                    )
                    session.add(recovery_case)
                    session.flush()
                elif event_type in RECOVERED_EVENTS and recovery_case is not None:
                    recovery_case.status = "recovered"

                recovery_case_id = recovery_case.id if recovery_case else None

            event.processed = True
            event.processed_at = datetime.now(UTC)
            event.processing_error = None
            session.commit()
        except Exception as error:
            session.rollback()
            event = session.scalar(
                select(WebhookEvent).where(
                    WebhookEvent.razorpay_event_id == event_id
                )
            )
            if event is not None:
                event.processing_error = str(error)[:2000]
                session.commit()
            raise

    return {
        "event_id": event_id,
        "event_type": event_type,
        "duplicate": False,
        "processed": True,
        "recovery_case_id": recovery_case_id,
    }
