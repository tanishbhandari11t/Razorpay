from __future__ import annotations

"""
Phase 14C outcome observer.

Watches pending shadow and controlled outcomes against database payment state.
Never enables provider actions. Never converts observational captures into
attributed intervention labels.
"""

from datetime import UTC, datetime

from app.database.connection import get_session
from app.services.outcome_observation import observe_pending_outcomes
from app.workers.celery_app import celery_app


@celery_app.task(name="recoverai.observe_pending_outcomes")
def observe_pending_outcome_windows(
    now_iso: str | None = None,
) -> dict[str, int]:
    observed_at = (
        datetime.fromisoformat(now_iso) if now_iso else datetime.now(UTC)
    )
    if observed_at.tzinfo is None:
        observed_at = observed_at.replace(tzinfo=UTC)
    with get_session() as session:
        result = observe_pending_outcomes(session, now=observed_at)
        session.commit()
    return result


celery_app.conf.beat_schedule = {
    **(celery_app.conf.beat_schedule or {}),
    "observe-pending-outcomes": {
        "task": "recoverai.observe_pending_outcomes",
        "schedule": 900.0,
    },
}
