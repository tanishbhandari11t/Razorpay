from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from kombu.exceptions import OperationalError as BrokerOperationalError
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError

from app.config.settings import get_settings
from app.database.connection import get_session
from app.models.recovery_job import RecoveryJob
from app.services.recovery_inference_v2_online import load_v2_shadow_config


def v2_online_automatic_enqueue_enabled() -> bool:
    config = load_v2_shadow_config()
    return bool(config["enabled"] and config["automatic_enqueue"])


def _identity() -> dict[str, str]:
    return load_v2_shadow_config()["identity"]


def _job_key(recovery_case_id: str) -> str:
    identity = _identity()
    return (
        f"{recovery_case_id}:{identity['task_name']}:"
        f"{identity['model_version']}:{identity['policy_version']}:"
        f"{identity['execution_mode']}"
    )


def ensure_v2_online_job(
    recovery_case_id: str,
) -> tuple[RecoveryJob, bool]:
    config = load_v2_shadow_config()
    if not config["enabled"]:
        raise RuntimeError("V2-online automatic challenger lane is disabled")
    identity = config["identity"]
    key = _job_key(recovery_case_id)
    with get_session() as session:
        existing = session.scalar(
            select(RecoveryJob).where(RecoveryJob.job_key == key)
        )
        if existing is not None:
            return existing, True
        job = RecoveryJob(
            recovery_case_id=recovery_case_id,
            job_key=key,
            task_name=identity["task_name"],
            model_version=identity["model_version"],
            policy_version=identity["policy_version"],
            execution_mode=identity["execution_mode"],
            status="queued",
            max_attempts=int(config["worker"]["max_attempts"]),
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


def publish_v2_online_job(job: RecoveryJob) -> dict[str, Any]:
    if job.status in {"succeeded", "permanent_failure", "failed"}:
        return {
            "job_id": job.id,
            "status": job.status,
            "published": False,
            "duplicate": True,
        }
    from app.workers.recovery_tasks_v2_online import (
        process_recovery_case_v2_online,
    )

    process_recovery_case_v2_online.app.conf.task_always_eager = (
        get_settings().celery_task_always_eager
    )
    try:
        result = process_recovery_case_v2_online.apply_async(
            args=[job.recovery_case_id, job.id]
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


def enqueue_v2_online_case(recovery_case_id: str) -> dict[str, Any]:
    job, duplicate = ensure_v2_online_job(recovery_case_id)
    if get_settings().celery_task_always_eager:
        result = publish_v2_online_job(job)
        result["duplicate"] = duplicate
        return result
    return {
        "job_id": job.id,
        "status": job.status,
        "published": False,
        "duplicate": duplicate,
    }


def dispatch_pending_v2_online_jobs(limit: int = 100) -> int:
    if not load_v2_shadow_config()["enabled"]:
        return 0
    task_name = _identity()["task_name"]
    with get_session() as session:
        jobs = session.scalars(
            select(RecoveryJob)
            .where(
                RecoveryJob.task_name == task_name,
                RecoveryJob.status.in_({"queued", "retry_pending"}),
            )
            .order_by(RecoveryJob.queued_at.asc())
            .limit(limit)
        ).all()
    return sum(bool(publish_v2_online_job(job)["published"]) for job in jobs)
