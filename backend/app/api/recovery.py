from __future__ import annotations

from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.database.connection import get_session
from app.ml.model_loader import ArtifactValidationError
from app.models.recovery_job import RecoveryJob
from app.services.features.customer_history import OnlineFeatureUnavailable
from app.services.payment_service import recent_recovery_cases
from app.services.recovery_engine import (
    RecoveryEngineError,
    persist_dry_run_policy_decision,
    recent_agent_decisions,
)
from app.services.recovery_inference import (
    ShadowInferenceError,
    evaluate_shadow_case,
    serialize_shadow_decision,
)
from app.services.recovery_inference_v2_online import (
    evaluate_v2_online_shadow_case,
    load_v2_shadow_config,
    serialize_v2_online_shadow_decision,
)
from app.services.recovery_jobs_v2_online import enqueue_v2_online_case
from app.services.shadow_ab_comparison import (
    compare_case_decisions,
    shadow_ab_metrics,
)
from app.services.shadow_monitoring import shadow_metrics
from app.services.shadow_evaluation_gate import evaluate_shadow_gate
from app.services.runtime_health import runtime_health


router = APIRouter(prefix="/api/recovery", tags=["recovery"])


class DryRunDecisionRequest(BaseModel):
    decision_key: str = Field(min_length=8, max_length=128)
    model_version: str = Field(default="recovery_model_v1", max_length=64)
    policy_version: str = Field(default="recovery_policy_v3", max_length=64)
    policy_manifest_sha256: str = Field(
        min_length=64,
        max_length=64,
    )
    decision_type: Literal["allow", "block", "fallback", "stop"]
    selected_action: Literal[
        "retry_payment",
        "payment_link",
        "whatsapp_reminder",
        "escalate_to_merchant",
    ] | None = None
    candidate_actions: dict[str, Any]
    predicted_probabilities: dict[str, float]
    expected_values: dict[str, float]
    decision_reasons: list[str]
    fallback_used: bool = False
    risk_checks: dict[str, Any]
    dry_run: Literal[True] = True


@router.get("/cases")
def recovery_cases(limit: int = 25) -> list[dict]:
    with get_session() as session:
        return recent_recovery_cases(session, max(1, min(limit, 100)))


@router.get("/shadow/metrics")
def get_shadow_metrics() -> dict[str, Any]:
    with get_session() as session:
        return shadow_metrics(session)


@router.get("/shadow/gate")
def get_shadow_gate() -> dict[str, Any]:
    with get_session() as session:
        return evaluate_shadow_gate(session)


@router.get("/shadow/v2-online/status")
def get_v2_online_shadow_status() -> dict[str, Any]:
    config = load_v2_shadow_config()
    return {
        "lane": config["lane"],
        "enabled": config["enabled"],
        "automatic_enqueue": config["automatic_enqueue"],
        "manual_shadow_evaluation": config[
            "manual_shadow_evaluation"
        ],
        "execution_mode": config["identity"]["execution_mode"],
        "provider_actions_enabled": config["safety"][
            "provider_actions_enabled"
        ],
    }


@router.get("/shadow/ab/metrics")
def get_shadow_ab_metrics() -> dict[str, Any]:
    with get_session() as session:
        return shadow_ab_metrics(session)


@router.post("/shadow/v2-online/{recovery_case_id}/evaluate")
def evaluate_v2_online_case(recovery_case_id: str) -> dict[str, Any]:
    config = load_v2_shadow_config()
    if not config["manual_shadow_evaluation"]:
        raise HTTPException(
            status_code=403,
            detail="V2-online manual shadow evaluation is disabled",
        )
    with get_session() as session:
        try:
            decision, duplicate = evaluate_v2_online_shadow_case(
                session,
                recovery_case_id,
            )
            session.commit()
            return serialize_v2_online_shadow_decision(
                decision,
                idempotent_replay=duplicate,
            )
        except ArtifactValidationError as error:
            session.rollback()
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (OnlineFeatureUnavailable, ShadowInferenceError) as error:
            session.rollback()
            status = 404 if str(error) == "Recovery case not found" else 409
            raise HTTPException(status_code=status, detail=str(error)) from error


@router.post("/shadow/v2-online/{recovery_case_id}/enqueue")
def enqueue_v2_online_shadow_case(
    recovery_case_id: str,
) -> dict[str, Any]:
    try:
        return enqueue_v2_online_case(recovery_case_id)
    except RuntimeError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/worker/health")
def get_worker_health() -> dict[str, object]:
    return runtime_health()


@router.get("/jobs")
def recovery_jobs(limit: int = 100) -> list[dict[str, Any]]:
    with get_session() as session:
        jobs = session.scalars(
            select(RecoveryJob)
            .order_by(RecoveryJob.queued_at.desc())
            .limit(max(1, min(limit, 500)))
        ).all()
        return [
            {
                "id": job.id,
                "recovery_case_id": job.recovery_case_id,
                "status": job.status,
                "attempts": job.attempts,
                "max_attempts": job.max_attempts,
                "error_class": job.error_class,
                "last_error": job.last_error,
                "celery_task_id": job.celery_task_id,
                "queued_at": job.queued_at.isoformat(),
                "started_at": (
                    job.started_at.isoformat() if job.started_at else None
                ),
                "completed_at": (
                    job.completed_at.isoformat()
                    if job.completed_at
                    else None
                ),
            }
            for job in jobs
        ]


@router.post("/{recovery_case_id}/evaluate")
def evaluate_recovery_case(recovery_case_id: str) -> dict[str, Any]:
    with get_session() as session:
        try:
            decision, duplicate = evaluate_shadow_case(
                session,
                recovery_case_id,
            )
            session.commit()
            return serialize_shadow_decision(
                decision,
                idempotent_replay=duplicate,
            )
        except ArtifactValidationError as error:
            session.rollback()
            raise HTTPException(status_code=503, detail=str(error)) from error
        except (OnlineFeatureUnavailable, ShadowInferenceError) as error:
            session.rollback()
            status = 404 if str(error) == "Recovery case not found" else 409
            raise HTTPException(status_code=status, detail=str(error)) from error


@router.post("/cases/{recovery_case_id}/decisions/dry-run")
def save_dry_run_decision(
    recovery_case_id: str,
    request: DryRunDecisionRequest,
) -> dict[str, Any]:
    with get_session() as session:
        try:
            decision, duplicate = persist_dry_run_policy_decision(
                session,
                recovery_case_id=recovery_case_id,
                **request.model_dump(),
            )
            session.commit()
        except RecoveryEngineError as error:
            session.rollback()
            status = 404 if str(error) == "Recovery case not found" else 400
            raise HTTPException(status_code=status, detail=str(error)) from error
    return {
        "decision_id": decision.id,
        "duplicate": duplicate,
        "dry_run": True,
        "selected_action": decision.selected_action,
        "execution_status": (
            "would_execute" if decision.selected_action else "no_action"
        ),
    }


@router.get("/cases/{recovery_case_id}/decisions")
def agent_decisions(recovery_case_id: str) -> list[dict[str, Any]]:
    with get_session() as session:
        return recent_agent_decisions(session, recovery_case_id)


@router.get("/cases/{recovery_case_id}/decisions/compare")
def compare_decisions(recovery_case_id: str) -> dict[str, Any]:
    with get_session() as session:
        return compare_case_decisions(session, recovery_case_id)


@router.post("/cases/{recovery_case_id}/execute")
def execute_recovery_case(recovery_case_id: str) -> dict[str, Any]:
    """Fail-closed controlled executor. Currently expected to BLOCK."""
    from app.services.recovery_executor import execute_approved_action

    with get_session() as session:
        result = execute_approved_action(session, case_id=recovery_case_id)
        session.commit()
        payload = result.as_dict()
        # Hard invariant for the final phase until evidence unlocks pilot.
        payload["executed"] = False
        return payload
