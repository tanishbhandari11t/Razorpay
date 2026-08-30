from __future__ import annotations

import time
from datetime import UTC, datetime

from celery import Task

from app.database.connection import get_session
from app.models.recovery_job import RecoveryJob
from app.workers.celery_app import celery_app
from app.workers import outcome_tasks as _outcome_tasks  # noqa: F401


def _start_attempt(job_id: str) -> tuple[int, int]:
    with get_session() as session:
        job = session.get(RecoveryJob, job_id)
        if job is None:
            raise ValueError("Reliability probe job not found")
        if job.status == "succeeded":
            return job.attempts, job.max_attempts
        job.status = "running"
        job.attempts += 1
        job.started_at = datetime.now(UTC)
        job.updated_at = datetime.now(UTC)
        session.commit()
        return job.attempts, job.max_attempts


def _finish(job_id: str, status: str, error: str | None = None) -> None:
    with get_session() as session:
        job = session.get(RecoveryJob, job_id)
        if job is None:
            raise ValueError("Reliability probe job not found")
        job.status = status
        job.last_error = error
        job.error_class = "TemporaryProbeFailure" if error else None
        job.updated_at = datetime.now(UTC)
        if status in {"succeeded", "failed"}:
            job.completed_at = datetime.now(UTC)
        session.commit()


@celery_app.task(
    bind=True,
    name="recoverai.retry_probe",
    acks_late=True,
    reject_on_worker_lost=True,
)
def retry_probe(
    self: Task,
    job_id: str,
    failures_before_success: int = 1,
) -> dict[str, object]:
    attempt, maximum = _start_attempt(job_id)
    if attempt <= failures_before_success:
        _finish(job_id, "retry_pending", "intentional temporary failure")
        raise self.retry(
            exc=ConnectionError("intentional temporary failure"),
            countdown=1,
            max_retries=maximum - 1,
        )
    _finish(job_id, "succeeded")
    return {"job_id": job_id, "status": "succeeded", "attempts": attempt}


@celery_app.task(
    bind=True,
    name="recoverai.crash_recovery_probe",
    acks_late=True,
    reject_on_worker_lost=True,
)
def crash_recovery_probe(
    self: Task,
    job_id: str,
    first_attempt_sleep_seconds: int = 60,
) -> dict[str, object]:
    del self
    attempt, maximum = _start_attempt(job_id)
    if attempt > maximum:
        _finish(job_id, "failed", "retry budget exhausted")
        return {"job_id": job_id, "status": "failed", "attempts": attempt}
    if attempt == 1:
        time.sleep(first_attempt_sleep_seconds)
    _finish(job_id, "succeeded")
    return {"job_id": job_id, "status": "succeeded", "attempts": attempt}
