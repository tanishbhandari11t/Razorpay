from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.intervention_outcome import InterventionOutcome
from app.models.outcome_observation import OutcomeObservation
from app.models.payment import Payment
from app.services.outcome_state_machine import (
    _as_utc,
    serialize_outcome,
    transition_persisted_outcome,
)
from ml.src.observe_recovery import (
    capture_attribution,
    checkpoint_status_updates,
    outcome_label_kind,
    time_to_recovery_seconds,
)


RECOVERED_STATUSES = {"captured", "paid", "authorized"}
TERMINAL_STATES = {"recovered", "no_recovery_observed", "unknown"}


def checkpoint_anchor(outcome: InterventionOutcome) -> datetime:
    if outcome.attempted_at is not None:
        return _as_utc(outcome.attempted_at)
    if outcome.failure_timestamp is not None:
        return _as_utc(outcome.failure_timestamp)
    return _as_utc(outcome.observation_window_starts_at)


def apply_status_checkpoints(
    outcome: InterventionOutcome,
    *,
    payment_status: str,
    observed_at: datetime,
) -> dict[str, str]:
    updates = checkpoint_status_updates(
        anchor_at=checkpoint_anchor(outcome),
        observed_at=_as_utc(observed_at),
        payment_status=payment_status,
        status_after_24h=outcome.payment_status_after_24h,
        status_after_48h=outcome.payment_status_after_48h,
    )
    if "payment_status_after_24h" in updates:
        outcome.payment_status_after_24h = updates["payment_status_after_24h"]
    if "payment_status_after_48h" in updates:
        outcome.payment_status_after_48h = updates["payment_status_after_48h"]
    return updates


def record_payment_observation(
    session: Session,
    *,
    payment: Payment,
    observation_source: str,
    external_ref: str,
    observed_at: datetime,
    payload: dict[str, Any],
) -> dict[str, int]:
    observed_at = _as_utc(observed_at)
    status = payment.status.lower()
    recovered_signal = status in RECOVERED_STATUSES
    outcomes = session.scalars(
        select(InterventionOutcome).where(
            InterventionOutcome.payment_id == payment.id
        )
    ).all()
    inserted = 0
    recovered = 0
    natural = 0
    for outcome in outcomes:
        existing = session.scalar(
            select(OutcomeObservation).where(
                OutcomeObservation.intervention_outcome_id == outcome.id,
                OutcomeObservation.observation_source == observation_source,
                OutcomeObservation.external_ref == external_ref,
            )
        )
        if existing is not None:
            continue
        starts_at = _as_utc(outcome.observation_window_starts_at)
        ends_at = _as_utc(outcome.observation_window_ends_at)
        attempted_at = (
            _as_utc(outcome.attempted_at) if outcome.attempted_at else None
        )
        attribution = capture_attribution(
            payment_status=status,
            attempted=outcome.attempted,
            attempted_at=attempted_at,
            observed_at=observed_at,
            window_starts_at=starts_at,
            window_ends_at=ends_at,
            outcome_state=outcome.outcome_state,
        )
        attribution_eligible = attribution == "attributed"
        observation = OutcomeObservation(
            intervention_outcome_id=outcome.id,
            observation_source=observation_source,
            external_ref=external_ref,
            payment_status=status,
            recovered_signal=recovered_signal,
            attribution_eligible=attribution_eligible,
            payload=payload,
            observed_at=observed_at,
        )
        session.add(observation)
        inserted += 1
        outcome.last_observed_at = observed_at
        apply_status_checkpoints(
            outcome,
            payment_status=status,
            observed_at=observed_at,
        )
        if recovered_signal and attribution_eligible:
            transition_persisted_outcome(
                outcome,
                "recovered",
                observed_at=observed_at,
                source=observation_source,
                reason="captured_payment_observed_after_attempt",
            )
            outcome.payment_recovered = True
            outcome.recovered_amount_minor = int(payment.amount)
            outcome.recovery_timestamp = observed_at
            outcome.time_to_recovery_seconds = time_to_recovery_seconds(
                failure_timestamp=(
                    _as_utc(outcome.failure_timestamp)
                    if outcome.failure_timestamp is not None
                    else attempted_at
                ),
                recovery_timestamp=observed_at,
            )
            recovered += 1
        elif attribution == "observational":
            outcome.natural_recovery_observed = True
            # Observational capture is evidence of independent repayment, not
            # intervention uplift. Never set payment_recovered=True here.
            if outcome.recovery_timestamp is None:
                outcome.recovery_timestamp = observed_at
                failure_at = (
                    _as_utc(outcome.failure_timestamp)
                    if outcome.failure_timestamp is not None
                    else starts_at
                )
                outcome.time_to_recovery_seconds = time_to_recovery_seconds(
                    failure_timestamp=failure_at,
                    recovery_timestamp=observed_at,
                )
                outcome.recovered_amount_minor = int(payment.amount)
            natural += 1
    session.flush()
    return {
        "observations_inserted": inserted,
        "attributed_recoveries": recovered,
        "natural_recoveries_observed": natural,
    }


def snapshot_due_checkpoints(
    session: Session,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    observed_at = _as_utc(now or datetime.now(UTC))
    outcomes = session.scalars(select(InterventionOutcome)).all()
    filled_24h = 0
    filled_48h = 0
    for outcome in outcomes:
        payment = session.get(Payment, outcome.payment_id)
        if payment is None:
            continue
        pending = checkpoint_status_updates(
            anchor_at=checkpoint_anchor(outcome),
            observed_at=observed_at,
            payment_status=payment.status.lower(),
            status_after_24h=outcome.payment_status_after_24h,
            status_after_48h=outcome.payment_status_after_48h,
        )
        if not pending:
            continue
        hours = 48 if "payment_status_after_48h" in pending else 24
        record_payment_observation(
            session,
            payment=payment,
            observation_source="database",
            external_ref=f"checkpoint:{hours}h:{outcome.id}",
            observed_at=observed_at,
            payload={"checkpoint_hours": hours, "poll": "database_status"},
        )
        if "payment_status_after_24h" in pending:
            filled_24h += 1
        if "payment_status_after_48h" in pending:
            filled_48h += 1
    session.flush()
    return {"filled_24h": filled_24h, "filled_48h": filled_48h}


def finalize_due_outcomes(
    session: Session,
    *,
    now: datetime | None = None,
) -> int:
    observed_at = _as_utc(now or datetime.now(UTC))
    session.flush()
    due = session.scalars(
        select(InterventionOutcome).where(
            InterventionOutcome.outcome_state == "waiting_for_outcome",
            InterventionOutcome.observation_window_ends_at <= observed_at,
        )
    ).all()
    for outcome in due:
        transition_persisted_outcome(
            outcome,
            "no_recovery_observed",
            observed_at=observed_at,
            source="database",
            reason="observation_window_elapsed_without_capture",
        )
        outcome.payment_recovered = None
        if not outcome.natural_recovery_observed:
            outcome.recovered_amount_minor = 0
    session.flush()
    return len(due)


def finalize_shadow_observation_windows(
    session: Session,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """Close shadow observation windows without attributing interventions."""
    observed_at = _as_utc(now or datetime.now(UTC))
    due = session.scalars(
        select(InterventionOutcome).where(
            InterventionOutcome.outcome_state == "decided",
            InterventionOutcome.attempted.is_(False),
            InterventionOutcome.observation_window_ends_at <= observed_at,
        )
    ).all()
    closed_no_recovery = 0
    closed_observational = 0
    closed_unknown = 0
    for outcome in due:
        payment = session.get(Payment, outcome.payment_id)
        status = (payment.status if payment is not None else "").lower()
        if outcome.natural_recovery_observed:
            # Independent capture already recorded; close as unknown terminal
            # only when status cannot be reconciled, else keep observational
            # provenance via natural_recovery_observed + outcome_at.
            outcome.outcome_at = observed_at
            outcome.last_observed_at = observed_at
            outcome.outcome_source = "database"
            outcome.state_history = [
                *(outcome.state_history or []),
                {
                    "previous_state": "decided",
                    "next_state": "decided",
                    "observed_at": observed_at.isoformat(),
                    "source": "database",
                    "reason": "shadow_observation_window_closed_with_independent_capture",
                },
            ]
            closed_observational += 1
            continue
        if status in {"", "created"} or payment is None:
            transition_persisted_outcome(
                outcome,
                "unknown",
                observed_at=observed_at,
                source="database",
                reason="shadow_payment_state_undetermined",
            )
            outcome.payment_recovered = None
            closed_unknown += 1
            continue
        transition_persisted_outcome(
            outcome,
            "no_recovery_observed",
            observed_at=observed_at,
            source="database",
            reason="shadow_observation_window_elapsed_without_capture",
        )
        outcome.payment_recovered = None
        outcome.recovered_amount_minor = 0
        closed_no_recovery += 1
    session.flush()
    return {
        "closed_no_recovery": closed_no_recovery,
        "closed_observational": closed_observational,
        "closed_unknown": closed_unknown,
    }


def observe_pending_outcomes(
    session: Session,
    *,
    now: datetime | None = None,
) -> dict[str, int]:
    """
    Phase 14C observer: inspect open outcomes against DB payment state,
    fill 24h/48h checkpoints, and close due windows. Does not call Razorpay
    providers and never enables attributed intervention labels in shadow.
    """
    observed_at = _as_utc(now or datetime.now(UTC))
    open_states = {"decided", "executed", "waiting_for_outcome"}
    outcomes = session.scalars(
        select(InterventionOutcome).where(
            InterventionOutcome.outcome_state.in_(open_states)
        )
    ).all()
    inspected = 0
    for outcome in outcomes:
        payment = session.get(Payment, outcome.payment_id)
        if payment is None:
            continue
        inspected += 1
        record_payment_observation(
            session,
            payment=payment,
            observation_source="database",
            external_ref=(
                f"observer:{outcome.agent_decision_id}:"
                f"{observed_at.strftime('%Y%m%d%H')}"
            ),
            observed_at=observed_at,
            payload={"observer": "pending_outcomes", "poll": "database_status"},
        )
    checkpoints = snapshot_due_checkpoints(session, now=observed_at)
    controlled_closed = finalize_due_outcomes(session, now=observed_at)
    shadow_closed = finalize_shadow_observation_windows(
        session,
        now=observed_at,
    )
    return {
        "inspected": inspected,
        "filled_24h": checkpoints["filled_24h"],
        "filled_48h": checkpoints["filled_48h"],
        "controlled_closed": controlled_closed,
        **shadow_closed,
    }


def list_case_outcomes(
    session: Session,
    recovery_case_id: str,
) -> list[dict[str, Any]]:
    outcomes = session.scalars(
        select(InterventionOutcome)
        .where(InterventionOutcome.recovery_case_id == recovery_case_id)
        .order_by(InterventionOutcome.created_at)
    ).all()
    return [serialize_outcome(outcome) for outcome in outcomes]


def outcome_metrics(session: Session) -> dict[str, Any]:
    outcomes = session.scalars(select(InterventionOutcome)).all()
    decisions = session.scalars(select(AgentDecision)).all()
    payments = session.scalars(select(Payment)).all()
    state_counts: dict[str, int] = {}
    source_counts: dict[str, int] = {}
    for outcome in outcomes:
        state_counts[outcome.outcome_state] = (
            state_counts.get(outcome.outcome_state, 0) + 1
        )
        source_counts[outcome.data_source] = (
            source_counts.get(outcome.data_source, 0) + 1
        )
    observations = int(
        session.scalar(select(func.count(OutcomeObservation.id))) or 0
    )
    attempted = sum(outcome.attempted for outcome in outcomes)
    attributed = sum(
        outcome.outcome_state == "recovered" and outcome.attempted
        for outcome in outcomes
    )
    observational = sum(
        outcome.natural_recovery_observed and not (
            outcome.outcome_state == "recovered" and outcome.attempted
        )
        for outcome in outcomes
    )
    eligible_labels = sum(
        outcome.outcome_state in {
            "recovered",
            "no_recovery_observed",
        }
        and outcome.attempted
        and outcome.data_source == "real_controlled"
        and not (
            outcome.natural_recovery_observed
            and outcome.outcome_state != "recovered"
        )
        for outcome in outcomes
    )
    return {
        "real_failures_observed": sum(
            payment.status.lower() == "failed" for payment in payments
        ),
        "shadow_decisions": sum(
            decision.execution_mode == "shadow" for decision in decisions
        ),
        "real_actions_executed": attempted,
        "observational_recoveries": observational,
        "attributed_intervention_recoveries": attributed,
        "training_eligible_labels": eligible_labels,
        "outcomes": len(outcomes),
        "observations": observations,
        "attempted_actions": attempted,
        "attributed_recoveries": attributed,
        "state_counts": state_counts,
        "data_source_counts": source_counts,
        "controlled_execution_authorized": False,
        "provider_actions_enabled": False,
        "phase15_authorized": False,
    }
