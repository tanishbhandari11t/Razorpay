from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from sqlalchemy import event
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.intervention_outcome import InterventionOutcome
from app.services.outcome_state_machine import (
    failure_timestamp_for_decision,
    new_outcome_for_decision,
)


@event.listens_for(Session, "before_flush")
def create_outcome_with_new_decision(
    session: Session,
    _flush_context,
    _instances,
) -> None:
    pending_outcome_decisions = {
        outcome.agent_decision_id
        for outcome in session.new
        if isinstance(outcome, InterventionOutcome)
    }
    for value in list(session.new):
        if not isinstance(value, AgentDecision):
            continue
        if value.id is None:
            value.id = str(uuid4())
        if value.created_at is None:
            value.created_at = datetime.now(UTC)
        if value.id in pending_outcome_decisions:
            continue
        session.add(
            new_outcome_for_decision(
                value,
                failure_timestamp=failure_timestamp_for_decision(
                    session,
                    value,
                ),
            )
        )
