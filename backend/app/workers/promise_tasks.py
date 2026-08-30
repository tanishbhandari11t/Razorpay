from __future__ import annotations

"""Celery beat: promise reminders (Trigger.dev-style delayed follow-ups)."""

from app.database.connection import get_session
from app.services.automation_state import load_automation_state
from app.services.promise_service import due_for_reminder, send_promise_reminder
from app.workers.celery_app import celery_app


@celery_app.task(name="recoverai.process_promise_reminders")
def process_promise_reminders() -> dict:
    state = load_automation_state()
    if not state.get("recover_revenue_enabled") or not state.get("promise_tracking_enabled"):
        return {"ok": True, "skipped": True, "reason": "automation_off"}
    with get_session() as session:
        due = due_for_reminder(session)
        sent = []
        for promise in due:
            result = send_promise_reminder(session, promise)
            sent.append(result)
        session.commit()
        return {"ok": True, "reminders": len(sent), "items": sent}
