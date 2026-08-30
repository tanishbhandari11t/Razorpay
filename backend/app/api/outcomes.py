from __future__ import annotations

from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlalchemy import select

from app.database.connection import get_session
from app.models.intervention_outcome import InterventionOutcome
from app.services.case_timeline import build_case_timeline
from app.services.communication_preview import build_communication_preview
from app.services.communication_templates import render_template
from app.services.controlled_pilot import assert_pilot_still_blocked
from app.services.inbox_service import needs_attention
from app.services.failure_gallery import failure_gallery, north_star_metrics
from app.services.decline_diagnoser import diagnose_failure
from app.services.dashboard_feeds import (
    agent_activity,
    evaluation_summary,
    intervention_stats,
    merchant_customers,
    recovery_queue,
)
from app.services.kill_switch import (
    kill_switch_status,
    set_kill_switch,
)
from app.services.outcome_observation import (
    list_case_outcomes,
    outcome_metrics,
)
from app.services.outcome_state_machine import serialize_outcome
from app.services.qwen_agent import validate_qwen_output
from app.services.recoverai_state import (
    evaluate_recoverai_state,
    progress_to_dict,
)
from app.services.communication_templates import list_supported_languages
from app.services.recovery_agent import draft_message_in_language, run_agent_for_case


router = APIRouter(prefix="/api/recovery", tags=["recovery-outcomes"])


class QwenValidateRequest(BaseModel):
    payload: dict[str, Any]
    policy_action: str | None = None
    taxonomy: str | None = None
    fraud: bool = False
    policy_allows_communication: bool = True


class CommunicationPreviewRequest(BaseModel):
    case_id: str | None = None
    policy_action: str | None = None
    amount_minor: int | None = None
    language: str = "english"
    customer_name: str | None = None
    taxonomy: str | None = None
    fraud: bool = False
    execution_mode: str = "shadow"


class LanguageDraftRequest(BaseModel):
    case_id: str
    language: str = "hinglish"
    template_action: str | None = Field(
        default=None,
        description="Optional template family: payment_link, whatsapp_reminder, etc.",
    )


@router.get("/outcomes/metrics")
def get_outcome_metrics() -> dict[str, Any]:
    with get_session() as session:
        metrics = outcome_metrics(session)
        outcomes = session.scalars(select(InterventionOutcome)).all()
        records = [serialize_outcome(outcome) for outcome in outcomes]
        progress = evaluate_recoverai_state(
            evidence_records=records,
            training_eligible_labels=metrics["training_eligible_labels"],
            shadow_decisions=metrics["shadow_decisions"],
            real_failures_observed=metrics["real_failures_observed"],
            execution_mode="shadow",
        )
    metrics["evidence_inventory"] = {
        "real_failures_observed": metrics["real_failures_observed"],
        "shadow_decisions": metrics["shadow_decisions"],
        "real_actions_executed": metrics["real_actions_executed"],
        "observational_recoveries": metrics["observational_recoveries"],
        "attributed_intervention_recoveries": metrics[
            "attributed_intervention_recoveries"
        ],
        "training_eligible_labels": metrics["training_eligible_labels"],
        "note": (
            "Shadow cases are not training examples. Independent captures "
            "are observational_recovery, not attributed_intervention_recovery."
        ),
    }
    metrics["phase15_authorized"] = False
    metrics["phase17_authorized"] = False
    metrics["recoverai_state"] = progress_to_dict(progress)
    pilot = assert_pilot_still_blocked()
    metrics["execution"] = {
        "mode": pilot.mode,
        "allowed": pilot.allowed,
        "reason": pilot.reason,
    }
    return metrics


@router.get("/state")
def get_recoverai_state() -> dict[str, Any]:
    with get_session() as session:
        metrics = outcome_metrics(session)
        outcomes = session.scalars(select(InterventionOutcome)).all()
        records = [serialize_outcome(outcome) for outcome in outcomes]
        progress = evaluate_recoverai_state(
            evidence_records=records,
            training_eligible_labels=metrics["training_eligible_labels"],
            shadow_decisions=metrics["shadow_decisions"],
            real_failures_observed=metrics["real_failures_observed"],
            execution_mode="shadow",
        )
    return progress_to_dict(progress)


@router.get("/cases/{recovery_case_id}/outcomes")
def get_case_outcomes(recovery_case_id: str) -> list[dict[str, Any]]:
    with get_session() as session:
        return list_case_outcomes(session, recovery_case_id)


@router.post("/agent/validate")
def validate_agent_output(request: QwenValidateRequest) -> dict[str, Any]:
    result = validate_qwen_output(
        request.payload,
        policy_action=request.policy_action,
        taxonomy=request.taxonomy,
        fraud=request.fraud,
        policy_allows_communication=request.policy_allows_communication,
    )
    fallback = None
    if not result.allowed and request.policy_action:
        fallback = render_template(
            request.policy_action,
            language=str(request.payload.get("language") or "english"),
        )
    return {
        "allowed": result.allowed,
        "intent": result.intent,
        "language": result.language,
        "message": result.message,
        "confidence": result.confidence,
        "reason": result.reason,
        "fallback_template": fallback,
        "financial_authority": False,
        "tools_enabled": False,
    }


@router.post("/agent/preview")
def preview_agent_communication(
    request: CommunicationPreviewRequest,
) -> dict[str, Any]:
    """Run the shadow recovery agent. Never sends or executes."""
    if request.case_id:
        with get_session() as session:
            result = run_agent_for_case(session, request.case_id)
            session.commit()
            return result
    if not request.policy_action or request.amount_minor is None:
        return {
            "ok": False,
            "executed": False,
            "blocked_reason": "case_id_or_policy_action_required",
            "execution_mode": "shadow",
        }
    return build_communication_preview(
        policy_action=request.policy_action,
        amount_minor=request.amount_minor,
        language=request.language,
        customer_name=request.customer_name,
        taxonomy=request.taxonomy,
        fraud=request.fraud,
        execution_mode=request.execution_mode,
    )


@router.get("/agent/languages")
def get_agent_languages() -> dict[str, Any]:
    return {
        "supported": list_supported_languages(),
        "default": "hinglish",
        "preview_only": True,
        "send_to_customer": False,
        "note": "Merchant can draft customer messages in Indian languages. Sending stays blocked in shadow mode.",
    }


@router.post("/agent/draft-language")
def draft_agent_language(request: LanguageDraftRequest) -> dict[str, Any]:
    """Draft / re-draft a customer message in the chosen Indian language."""
    with get_session() as session:
        result = draft_message_in_language(
            session,
            recovery_case_id=request.case_id,
            language=request.language,
            template_action=request.template_action,
        )
        session.commit()
        return result


@router.get("/queue")
def get_recovery_queue(limit: int = 50) -> list[dict[str, Any]]:
    with get_session() as session:
        return recovery_queue(session, limit)


@router.get("/agent/activity")
def get_agent_activity(limit: int = 50) -> list[dict[str, Any]]:
    with get_session() as session:
        return agent_activity(session, limit)


@router.get("/evaluation")
def get_evaluation_summary() -> dict[str, Any]:
    with get_session() as session:
        return evaluation_summary(session)


@router.get("/inbox")
def get_needs_attention(limit: int = 20) -> dict[str, Any]:
    with get_session() as session:
        return needs_attention(session, limit=limit)


@router.get("/failure-gallery")
def get_failure_gallery(limit: int = 100) -> dict[str, Any]:
    """ReCoup-inspired failure class gallery for merchants."""
    with get_session() as session:
        return failure_gallery(session, limit=limit)


@router.get("/north-star")
def get_north_star() -> dict[str, Any]:
    """RecoverAI vs dumb baseline comparison (ReCoup north-star pattern)."""
    with get_session() as session:
        return north_star_metrics(session)


@router.get("/diagnose")
def diagnose_reason(reason: str = "") -> dict[str, Any]:
    return diagnose_failure(reason)


@router.get("/customers")
def get_merchant_customers(limit: int = 50) -> list[dict[str, Any]]:
    with get_session() as session:
        return merchant_customers(session, limit)


@router.get("/interventions/stats")
def get_intervention_stats() -> list[dict[str, Any]]:
    with get_session() as session:
        return intervention_stats(session)


@router.get("/cases/{recovery_case_id}/timeline")
def get_case_timeline(recovery_case_id: str) -> dict[str, Any]:
    with get_session() as session:
        return build_case_timeline(session, recovery_case_id)


class KillSwitchRequest(BaseModel):
    armed: bool = Field(description="True arms emergency stop")


@router.get("/kill-switch")
def get_kill_switch() -> dict[str, Any]:
    return kill_switch_status()


@router.post("/kill-switch")
def update_kill_switch(request: KillSwitchRequest) -> dict[str, Any]:
    return set_kill_switch(request.armed)

