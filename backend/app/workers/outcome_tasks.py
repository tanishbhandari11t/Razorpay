from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import select

from app.database.connection import get_session
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment
from app.services.outcome_observation import (
    finalize_due_outcomes,
    observe_pending_outcomes,
    record_payment_observation,
    snapshot_due_checkpoints,
)
from app.services.outcome_state_machine import load_real_outcome_schema
from app.services.razorpay_service import check_payment_status
from app.workers.celery_app import celery_app
from app.workers import outcome_observer as _outcome_observer  # noqa: F401


@celery_app.task(name="recoverai.finalize_due_outcomes")
def finalize_due_outcome_windows() -> dict[str, int]:
    with get_session() as session:
        finalized = finalize_due_outcomes(session)
        session.commit()
    return {"finalized": finalized}


@celery_app.task(name="recoverai.poll_open_outcomes")
def poll_open_outcomes() -> dict[str, int | bool]:
    config = load_real_outcome_schema()
    if not config["observation"]["provider_poll_enabled"]:
        return {"enabled": False, "polled": 0, "errors": 0}
    now = datetime.now(UTC)
    polled = 0
    errors = 0
    with get_session() as session:
        rows = session.execute(
            select(InterventionOutcome, Payment)
            .join(Payment, Payment.id == InterventionOutcome.payment_id)
            .where(
                InterventionOutcome.outcome_state
                == "waiting_for_outcome",
                InterventionOutcome.attempted.is_(True),
            )
        ).all()
        for outcome, payment in rows:
            if not payment.razorpay_payment_id:
                errors += 1
                continue
            try:
                payment.status = check_payment_status(
                    payment.razorpay_payment_id
                )
                record_payment_observation(
                    session,
                    payment=payment,
                    observation_source="provider",
                    external_ref=(
                        f"poll:{payment.razorpay_payment_id}:"
                        f"{now.strftime('%Y%m%d%H')}"
                    ),
                    observed_at=now,
                    payload={"poll": "payment.fetch"},
                )
                polled += 1
            except Exception:
                errors += 1
        session.commit()
    return {"enabled": True, "polled": polled, "errors": errors}


@celery_app.task(name="recoverai.snapshot_outcome_checkpoints")
def snapshot_outcome_checkpoints() -> dict[str, int]:
    with get_session() as session:
        filled = snapshot_due_checkpoints(session)
        session.commit()
    return filled


@celery_app.task(name="recoverai.run_outcome_observer")
def run_outcome_observer() -> dict[str, int]:
    with get_session() as session:
        result = observe_pending_outcomes(session)
        session.commit()
    return result


celery_app.conf.beat_schedule = {
    **(celery_app.conf.beat_schedule or {}),
    "finalize-due-outcome-windows": {
        "task": "recoverai.finalize_due_outcomes",
        "schedule": 300.0,
    },
    "poll-open-outcomes": {
        "task": "recoverai.poll_open_outcomes",
        "schedule": 3600.0,
    },
    "snapshot-outcome-checkpoints": {
        "task": "recoverai.snapshot_outcome_checkpoints",
        "schedule": 1800.0,
    },
    "run-outcome-observer": {
        "task": "recoverai.run_outcome_observer",
        "schedule": 900.0,
    },
}
