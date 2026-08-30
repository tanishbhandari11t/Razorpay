from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kombu.exceptions import OperationalError as BrokerOperationalError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config.settings import get_settings
from app.database.connection import get_session
from app.models.recovery_job import RecoveryJob
from app.services.execution_gate import load_execution_gate


TASK_NAME = "shadow_inference"
MODEL_VERSION = "recovery_model_v1"
POLICY_VERSION = "recovery_policy_v3"
EXECUTION_MODE = "shadow"


def _job_key(recovery_case_id: str) -> str:
    return (
        f"{recovery_case_id}:{TASK_NAME}:{MODEL_VERSION}:"
        f"{POLICY_VERSION}:{EXECUTION_MODE}"
    )


def ensure_recovery_job(recovery_case_id: str) -> tuple[RecoveryJob, bool]:
    key = _job_key(recovery_case_id)
    with get_session() as session:
        existing = session.scalar(
            select(RecoveryJob).where(RecoveryJob.job_key == key)
        )
        if existing is not None:
            return existing, True
        max_attempts = int(
            load_execution_gate()["worker"]["max_attempts"]
        )
        job = RecoveryJob(
            recovery_case_id=recovery_case_id,
            job_key=key,
            task_name=TASK_NAME,
            model_version=MODEL_VERSION,
            policy_version=POLICY_VERSION,
            execution_mode=EXECUTION_MODE,
            status="queued",
            max_attempts=max_attempts,
        )
        session.add(job)
        try:
            session.commit()
        except IntegrityError:
            session.rollback()
            existing = session.scalar(
                select(RecoveryJob).where(RecoveryJob.job_key == key)
            )
            if existing is None:
                raise
            return existing, True
        return job, False


def publish_recovery_job(job: RecoveryJob) -> dict[str, Any]:
    if job.status in {"succeeded", "permanent_failure", "failed"}:
        return {
            "job_id": job.id,
            "status": job.status,
            "published": False,
            "duplicate": True,
        }
    from app.workers.recovery_tasks import process_recovery_case

    process_recovery_case.app.conf.task_always_eager = (
        get_settings().celery_task_always_eager
    )
    try:
        result = process_recovery_case.apply_async(
            args=[job.recovery_case_id, job.id],
        )
    except (BrokerOperationalError, ConnectionError, OSError) as error:
        with get_session() as session:
            persisted = session.get(RecoveryJob, job.id)
            if persisted is not None and persisted.status == "queued":
                persisted.last_error = str(error)[:2000]
                persisted.error_class = type(error).__name__
                persisted.updated_at = datetime.now(UTC)
                session.commit()
        return {
            "job_id": job.id,
            "status": "queued",
            "published": False,
            "duplicate": False,
        }

    with get_session() as session:
        persisted = session.get(RecoveryJob, job.id)
        status = "queued"
        if persisted is not None:
            persisted.celery_task_id = result.id
            persisted.updated_at = datetime.now(UTC)
            session.commit()
            status = persisted.status
    return {
        "job_id": job.id,
        "status": status,
        "published": True,
        "duplicate": False,
    }


def enqueue_recovery_case(recovery_case_id: str) -> dict[str, Any]:
    job, duplicate = ensure_recovery_job(recovery_case_id)
    if get_settings().celery_task_always_eager:
        result = publish_recovery_job(job)
        result["duplicate"] = duplicate
        return result
    return {
        "job_id": job.id,
        "status": job.status,
        "published": False,
        "duplicate": duplicate,
    }


def dispatch_pending_recovery_jobs(limit: int = 100) -> int:
    with get_session() as session:
        jobs = session.scalars(
            select(RecoveryJob)
            .where(
                RecoveryJob.task_name == TASK_NAME,
                RecoveryJob.status.in_({"queued", "retry_pending"}),
            )
            .order_by(RecoveryJob.queued_at.asc())
            .limit(limit)
        ).all()
    return sum(bool(publish_recovery_job(job)["published"]) for job in jobs)
