from __future__ import annotations

from celery import Celery

from app.config.settings import get_settings


settings = get_settings()
celery_app = Celery(
    "recoverai",
    broker=settings.redis_url,
    backend=settings.redis_url,
    include=[
        "app.workers.recovery_tasks",
        "app.workers.recovery_tasks_v2_online",
        "app.workers.reliability_tasks",
        "app.workers.promise_tasks",
    ],
)
celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_acks_late=True,
    task_reject_on_worker_lost=True,
    task_track_started=True,
    worker_prefetch_multiplier=1,
    broker_connection_retry_on_startup=True,
    broker_connection_max_retries=3,
    broker_connection_timeout=2,
    task_publish_retry=False,
    broker_transport_options={"visibility_timeout": 30},
    result_backend_transport_options={"visibility_timeout": 30},
    task_always_eager=settings.celery_task_always_eager,
    task_eager_propagates=True,
    beat_schedule={
        "redispatch-pending-recovery-jobs": {
            "task": "recoverai.dispatch_pending_recovery_jobs",
            "schedule": 30.0,
        },
        "redispatch-pending-v2-online-jobs": {
            "task": "recoverai.dispatch_pending_v2_online_jobs",
            "schedule": 30.0,
        },
        "process-promise-reminders": {
            "task": "recoverai.process_promise_reminders",
            "schedule": 300.0,
        },
    },
)
