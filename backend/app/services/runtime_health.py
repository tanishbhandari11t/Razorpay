from __future__ import annotations

from sqlalchemy import func, select
from redis import Redis

from app.config.settings import get_settings
from app.database.connection import database_backend, get_session
from app.ml.model_loader import load_model_bundle
from app.models.recovery_job import RecoveryJob
from app.services.execution_gate import load_execution_gate
from app.workers.celery_app import celery_app


def runtime_health() -> dict[str, object]:
    settings = get_settings()
    redis_ready = False
    try:
        client = Redis.from_url(
            settings.redis_url,
            socket_connect_timeout=1,
            socket_timeout=1,
        )
        redis_ready = bool(client.ping())
    except Exception:
        redis_ready = False
    try:
        worker_replies = celery_app.control.ping(timeout=1)
    except Exception:
        worker_replies = []
    worker_ready = bool(worker_replies)
    bundle = load_model_bundle()
    gate = load_execution_gate()
    with get_session() as session:
        pending_jobs = int(
            session.scalar(
                select(func.count(RecoveryJob.id)).where(
                    RecoveryJob.task_name == "shadow_inference",
                    RecoveryJob.status.in_(
                        {"queued", "running", "retry_pending"}
                    ),
                )
            )
            or 0
        )
        failed_jobs = int(
            session.scalar(
                select(func.count(RecoveryJob.id)).where(
                    RecoveryJob.task_name == "shadow_inference",
                    RecoveryJob.status.in_(
                        {"failed", "permanent_failure"}
                    ),
                )
            )
            or 0
        )
    return {
        "database": database_backend(),
        "redis": "available" if redis_ready else "unavailable",
        "worker": "ready" if worker_ready else "unavailable",
        "worker_nodes": len(worker_replies),
        "pending_jobs": pending_jobs,
        "failed_jobs": failed_jobs,
        "model": bundle.model_version,
        "policy": bundle.policy_version,
        "feature_count": len(bundle.raw_feature_names),
        "execution": gate["execution"]["mode"],
        "provider_actions_enabled": bool(
            gate["execution"]["provider_actions_enabled"]
        ),
    }
