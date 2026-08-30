from __future__ import annotations

from datetime import UTC, datetime

from celery import Task
from sqlalchemy.exc import OperationalError

from app.database.connection import get_session
from app.ml.model_loader import ArtifactValidationError
from app.models.recovery_job import RecoveryJob
from app.services.execution_gate import load_execution_gate
from app.services.features.customer_history import OnlineFeatureUnavailable
from app.services.recovery_inference import (
    ShadowInferenceError,
    evaluate_shadow_case,
)
from app.services.recovery_jobs import dispatch_pending_recovery_jobs
from app.workers.celery_app import celery_app


RETRYABLE_ERRORS = (
    OperationalError,
    ConnectionError,
    TimeoutError,
)
PERMANENT_ERRORS = (
    ArtifactValidationError,
    OnlineFeatureUnavailable,
    ShadowInferenceError,
    ValueError,
)


def _retry_countdown(attempt: int) -> int:
    backoffs = load_execution_gate()["worker"]["retry_backoff_seconds"]
    return int(backoffs[min(max(attempt - 1, 0), len(backoffs) - 1)])


def _finish_job(
    job_id: str,
    *,
    status: str,
    error: BaseException | None = None,
) -> None:
    with get_session() as session:
        job = session.get(RecoveryJob, job_id)
        if job is None:
            return
        job.status = status
        job.last_error = str(error)[:2000] if error else None
        job.error_class = type(error).__name__ if error else None
        job.updated_at = datetime.now(UTC)
        if status in {"succeeded", "failed", "permanent_failure"}:
            job.completed_at = datetime.now(UTC)
        session.commit()


@celery_app.task(
    bind=True,
    name="recoverai.process_recovery_case",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_recovery_case(
    self: Task,
    recovery_case_id: str,
    job_id: str,
) -> dict[str, object]:
    try:
        with get_session() as session:
            job = session.get(RecoveryJob, job_id)
            if job is None:
                raise ShadowInferenceError("Recovery job not found")
            if job.recovery_case_id != recovery_case_id:
                raise ShadowInferenceError("Recovery job identity mismatch")
            if job.status == "succeeded":
                return {
                    "job_id": job.id,
                    "status": "succeeded",
                    "idempotent_replay": True,
                }
            if job.status in {"failed", "permanent_failure"}:
                return {
                    "job_id": job.id,
                    "status": job.status,
                    "idempotent_replay": True,
                }
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.error_class = "WorkerRetryBudgetExhausted"
                job.last_error = (
                    "Job was redelivered after the retry budget was exhausted"
                )
                job.completed_at = datetime.now(UTC)
                job.updated_at = datetime.now(UTC)
                session.commit()
                return {
                    "job_id": job.id,
                    "status": "failed",
                    "error_class": job.error_class,
                }
            job.status = "running"
            job.attempts += 1
            job.started_at = datetime.now(UTC)
            job.updated_at = datetime.now(UTC)
            attempt = job.attempts
            max_attempts = job.max_attempts
            session.commit()

        with get_session() as session:
            decision, duplicate = evaluate_shadow_case(
                session,
                recovery_case_id,
            )
            session.commit()
        _finish_job(job_id, status="succeeded")
        return {
            "job_id": job_id,
            "status": "succeeded",
            "decision_id": decision.id,
            "idempotent_replay": duplicate,
        }
    except PERMANENT_ERRORS as error:
        _finish_job(job_id, status="permanent_failure", error=error)
        return {
            "job_id": job_id,
            "status": "permanent_failure",
            "error_class": type(error).__name__,
        }
    except RETRYABLE_ERRORS as error:
        if "attempt" not in locals() or "max_attempts" not in locals():
            attempt = int(self.request.retries) + 1
            max_attempts = int(
                load_execution_gate()["worker"]["max_attempts"]
            )
        if attempt >= max_attempts:
            _finish_job(job_id, status="failed", error=error)
            return {
                "job_id": job_id,
                "status": "failed",
                "error_class": type(error).__name__,
            }
        _finish_job(job_id, status="retry_pending", error=error)
        raise self.retry(
            exc=error,
            countdown=_retry_countdown(attempt),
            max_retries=max_attempts - 1,
        )
    except Exception as error:
        _finish_job(job_id, status="permanent_failure", error=error)
        return {
            "job_id": job_id,
            "status": "permanent_failure",
            "error_class": type(error).__name__,
        }


@celery_app.task(name="recoverai.dispatch_pending_recovery_jobs")
def dispatch_pending_jobs() -> dict[str, int]:
    return {"published": dispatch_pending_recovery_jobs()}
