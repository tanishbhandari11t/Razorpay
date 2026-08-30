from __future__ import annotations

from datetime import UTC, datetime

from celery import Task

from app.database.connection import get_session
from app.models.recovery_job import RecoveryJob
from app.services.recovery_inference import ShadowInferenceError
from app.services.recovery_inference_v2_online import (
    evaluate_v2_online_shadow_case,
    load_v2_shadow_config,
)
from app.services.recovery_jobs_v2_online import (
    dispatch_pending_v2_online_jobs,
)
from app.workers.celery_app import celery_app
from app.workers.recovery_tasks import (
    PERMANENT_ERRORS,
    RETRYABLE_ERRORS,
    _finish_job,
)


def _retry_countdown(attempt: int) -> int:
    backoffs = load_v2_shadow_config()["worker"]["retry_backoff_seconds"]
    return int(backoffs[min(max(attempt - 1, 0), len(backoffs) - 1)])


@celery_app.task(
    bind=True,
    name="recoverai.process_recovery_case_v2_online",
    acks_late=True,
    reject_on_worker_lost=True,
)
def process_recovery_case_v2_online(
    self: Task,
    recovery_case_id: str,
    job_id: str,
) -> dict[str, object]:
    try:
        if not load_v2_shadow_config()["enabled"]:
            raise ShadowInferenceError(
                "V2-online challenger lane was disabled before execution"
            )
        with get_session() as session:
            job = session.get(RecoveryJob, job_id)
            if job is None:
                raise ShadowInferenceError("V2-online recovery job not found")
            if job.recovery_case_id != recovery_case_id:
                raise ShadowInferenceError(
                    "V2-online recovery job identity mismatch"
                )
            if job.status in {"succeeded", "failed", "permanent_failure"}:
                return {
                    "job_id": job.id,
                    "status": job.status,
                    "idempotent_replay": True,
                }
            if job.attempts >= job.max_attempts:
                job.status = "failed"
                job.error_class = "WorkerRetryBudgetExhausted"
                job.last_error = "V2-online retry budget exhausted"
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
            decision, duplicate = evaluate_v2_online_shadow_case(
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
                load_v2_shadow_config()["worker"]["max_attempts"]
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


@celery_app.task(name="recoverai.dispatch_pending_v2_online_jobs")
def dispatch_pending_jobs_v2_online() -> dict[str, int]:
    return {"published": dispatch_pending_v2_online_jobs()}
