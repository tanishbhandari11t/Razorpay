from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.intervention import Intervention
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.observe_recovery import outcome_label_kind, transition_outcome
from ml.src.record_recovery_outcome import load_real_outcome_schema


def _as_utc(value: datetime) -> datetime:
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value


def _data_source(execution_mode: str) -> str:
    if execution_mode == "controlled":
        return "real_controlled"
    if execution_mode == "shadow":
        return "real_shadow"
    return "synthetic"


def failure_timestamp_for_decision(
    session: Session,
    decision: AgentDecision,
) -> datetime:
    payment = session.get(Payment, decision.payment_id)
    if payment is None:
        for obj in list(session.new) + list(session.dirty):
            if isinstance(obj, Payment) and obj.id == decision.payment_id:
                payment = obj
                break
    if payment is not None and payment.created_at is not None:
        return _as_utc(payment.created_at)
    return _as_utc(decision.created_at or datetime.now(UTC))


def new_outcome_for_decision(
    decision: AgentDecision,
    *,
    failure_timestamp: datetime | None = None,
) -> InterventionOutcome:
    schema = load_real_outcome_schema()
    starts_at = _as_utc(decision.created_at or datetime.now(UTC))
    failed_at = _as_utc(failure_timestamp or starts_at)
    ends_at = starts_at + timedelta(
        hours=float(schema["observation"]["default_window_hours"])
    )
    probability = (
        float(decision.predicted_probabilities[decision.selected_action])
        if decision.selected_action
        and decision.selected_action in decision.predicted_probabilities
        else None
    )
    return InterventionOutcome(
        outcome_key=f"{decision.id}:outcome",
        agent_decision_id=decision.id,
        payment_id=decision.payment_id,
        recovery_case_id=decision.recovery_case_id,
        action=decision.selected_action,
        decision_probability=probability,
        decision_margin=decision.decision_margin,
        model_version=decision.model_version,
        policy_version=decision.policy_version,
        execution_mode=decision.execution_mode,
        attempted=False,
        failure_timestamp=failed_at,
        outcome_state="decided",
        payment_recovered=None,
        recovered_amount_minor=0,
        observation_window_starts_at=starts_at,
        observation_window_ends_at=ends_at,
        outcome_source="database",
        data_source=_data_source(decision.execution_mode),
        natural_recovery_observed=False,
        state_history=[
            {
                "previous_state": None,
                "next_state": "decided",
                "observed_at": starts_at.isoformat(),
                "source": "database",
                "reason": "agent_decision_persisted",
            }
        ],
    )


def create_outcome_for_decision(
    session: Session,
    decision: AgentDecision,
) -> tuple[InterventionOutcome, bool]:
    existing = session.scalar(
        select(InterventionOutcome).where(
            InterventionOutcome.agent_decision_id == decision.id
        )
    )
    if existing is not None:
        return existing, True
    outcome = new_outcome_for_decision(
        decision,
        failure_timestamp=failure_timestamp_for_decision(session, decision),
    )
    session.add(outcome)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.agent_decision_id == decision.id
            )
        )
        if existing is None:
            raise
        return existing, True
    return outcome, False


def transition_persisted_outcome(
    outcome: InterventionOutcome,
    next_state: str,
    *,
    observed_at: datetime,
    source: str,
    reason: str,
) -> None:
    observed_at = _as_utc(observed_at)
    transition = transition_outcome(
        outcome.outcome_state,
        next_state,
        observed_at=observed_at,
        source=source,
        reason=reason,
    )
    outcome.outcome_state = next_state
    outcome.outcome_source = source
    outcome.last_observed_at = observed_at
    outcome.state_history = [
        *(outcome.state_history or []),
        transition.to_dict(),
    ]
    if next_state in {"recovered", "no_recovery_observed", "unknown"}:
        outcome.outcome_at = observed_at


def mark_outcome_executed(
    outcome: InterventionOutcome,
    intervention: Intervention,
    *,
    attempted_at: datetime,
    source: str = "provider",
) -> None:
    attempted_at = _as_utc(attempted_at)
    if outcome.execution_mode != "controlled":
        raise ValueError(
            "Only controlled decisions may be marked as provider-executed"
        )
    if intervention.agent_decision_id != outcome.agent_decision_id:
        raise ValueError("Intervention and outcome decision IDs differ")
    outcome.intervention_id = intervention.id
    outcome.attempted = True
    outcome.attempted_at = attempted_at
    transition_persisted_outcome(
        outcome,
        "executed",
        observed_at=attempted_at,
        source=source,
        reason="provider_action_attempted",
    )
    transition_persisted_outcome(
        outcome,
        "waiting_for_outcome",
        observed_at=attempted_at,
        source=source,
        reason="observation_window_opened",
    )


def serialize_outcome(outcome: InterventionOutcome) -> dict[str, Any]:
    return {
        "id": outcome.id,
        "decision_id": outcome.agent_decision_id,
        "intervention_id": outcome.intervention_id,
        "payment_id": outcome.payment_id,
        "recovery_case_id": outcome.recovery_case_id,
        "action": outcome.action,
        "decision_probability": outcome.decision_probability,
        "decision_margin": outcome.decision_margin,
        "model_version": outcome.model_version,
        "policy_version": outcome.policy_version,
        "execution_mode": outcome.execution_mode,
        "attempted": outcome.attempted,
        "attempted_at": (
            outcome.attempted_at.isoformat()
            if outcome.attempted_at
            else None
        ),
        "failure_timestamp": (
            outcome.failure_timestamp.isoformat()
            if outcome.failure_timestamp
            else None
        ),
        "payment_status_after_24h": outcome.payment_status_after_24h,
        "payment_status_after_48h": outcome.payment_status_after_48h,
        "outcome_state": outcome.outcome_state,
        "outcome_at": (
            outcome.outcome_at.isoformat() if outcome.outcome_at else None
        ),
        "payment_recovered": outcome.payment_recovered,
        "recovered_amount_minor": outcome.recovered_amount_minor,
        "recovery_timestamp": (
            outcome.recovery_timestamp.isoformat()
            if outcome.recovery_timestamp
            else None
        ),
        "time_to_recovery_seconds": outcome.time_to_recovery_seconds,
        "observation_window_starts_at": (
            outcome.observation_window_starts_at.isoformat()
        ),
        "observation_window_ends_at": (
            outcome.observation_window_ends_at.isoformat()
        ),
        "outcome_source": outcome.outcome_source,
        "data_source": outcome.data_source,
        "natural_recovery_observed": outcome.natural_recovery_observed,
        "label_kind": outcome_label_kind(
            outcome_state=outcome.outcome_state,
            attempted=outcome.attempted,
            payment_recovered=outcome.payment_recovered,
            natural_recovery_observed=outcome.natural_recovery_observed,
            data_source=outcome.data_source,
        ),
        "state_history": outcome.state_history,
    }
