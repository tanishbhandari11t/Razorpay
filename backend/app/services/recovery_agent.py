from __future__ import annotations

"""
RecoverAI recovery agent — shadow / preview only.

Policy remains financial authority. This agent drafts communication,
validates it, and returns an executable plan that is never executed.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, ValidationError
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.communication_preview import build_communication_preview
from app.services.communication_templates import (
    normalize_language,
    render_template,
)
from app.services.qwen_agent import validate_qwen_output


REPO_ROOT = Path(__file__).resolve().parents[3]
ACTIONS_PATH = REPO_ROOT / "ml" / "config" / "agent_actions.yaml"

SYSTEM_PROMPT = """You are RecoverAI's communication assistant.

Your ONLY job is to draft a customer-facing message.

You MUST:
- follow the supplied action exactly
- preserve the supplied amount
- use the supplied language
- be concise
- never invent payment status
- never invent discounts/refunds
- never request sensitive credentials
- never change the recommended action

You MUST NOT:
- choose an intervention
- authorize a transaction
- call APIs
- execute payments
- modify financial data

Return only the requested structured response.
"""


class AgentMessage(BaseModel):
    language: str = "english"
    message: str = Field(min_length=1, max_length=800)


class AgentCaseInput(BaseModel):
    case_id: str
    amount_inr: float
    amount_minor: int
    action: str
    failure_category: str | None = None
    language: str = "english"
    customer_segment: str = "unknown"
    customer_name: str | None = None
    fraud: bool = False
    recovery_probability: float | None = None
    decision_type: str | None = None
    policy_version: str | None = None
    model_version: str | None = None


@dataclass(frozen=True)
class RecoveryAgentResult:
    status: str
    action: str | None
    message: str | None
    language: str | None
    executed: bool
    execution_mode: str
    communication_model: str
    communication_status: str
    blocked_reason: str | None
    recovery_probability: float | None
    case_id: str
    payload: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "status": self.status,
            "action": self.action,
            "message": self.message,
            "language": self.language,
            "executed": False,
            "execution_mode": self.execution_mode,
            "communication_model": self.communication_model,
            "communication_status": self.communication_status,
            "blocked_reason": self.blocked_reason,
            "recovery_probability": self.recovery_probability,
            "financial_authority": False,
            "qwen_tools_enabled": False,
            "system_prompt_applied": True,
            "agent_input": self.payload,
        }


@lru_cache(maxsize=1)
def load_agent_actions() -> dict[str, Any]:
    return yaml.safe_load(ACTIONS_PATH.read_text(encoding="utf-8"))


def _normalize_language(value: str | None) -> str:
    return normalize_language(value)


def _action_allows_communication(action: str | None) -> bool:
    config = load_agent_actions()
    if not action:
        return False
    entry = config.get("actions", {}).get(action, {})
    return bool(entry.get("communication", config["defaults"]["communication"]))


def _extract_taxonomy(risk_checks: Any) -> str | None:
    if not isinstance(risk_checks, list):
        return None
    for check in risk_checks:
        if isinstance(check, dict) and check.get("name") == "phase12_compatibility_gate":
            return check.get("failure_taxonomy")
    return None


def _selected_probability(decision: AgentDecision) -> float | None:
    probs = decision.predicted_probabilities or {}
    action = decision.selected_action
    if action and action in probs:
        try:
            return float(probs[action])
        except (TypeError, ValueError):
            return None
    if probs:
        try:
            return float(max(probs.values()))
        except (TypeError, ValueError):
            return None
    return None


def build_agent_input(
    *,
    recovery_case: RecoveryCase,
    payment: Payment,
    customer: Customer,
    decision: AgentDecision,
) -> AgentCaseInput:
    amount_minor = int(payment.amount)
    segment = (
        "returning"
        if int((decision.features_snapshot or {}).get("has_prior_history") or 0) == 1
        else "new"
    )
    return AgentCaseInput(
        case_id=recovery_case.id,
        amount_inr=round(amount_minor / 100, 2),
        amount_minor=amount_minor,
        action=decision.selected_action or "escalate_to_merchant",
        failure_category=_extract_taxonomy(decision.risk_checks)
        or decision.failure_class,
        language=_normalize_language(customer.preferred_language),  # type: ignore[arg-type]
        customer_segment=segment,
        customer_name=customer.name,
        fraud=bool(int((decision.features_snapshot or {}).get("fraud_flag") or 0)),
        recovery_probability=_selected_probability(decision),
        decision_type=decision.decision_type,
        policy_version=decision.policy_version,
        model_version=decision.model_version,
    )


def validate_structured_message(
    raw: dict[str, Any],
    *,
    policy_action: str,
    amount_minor: int,
    allowed_payment_link: str | None = None,
) -> tuple[AgentMessage | None, str | None]:
    if "action" in raw and str(raw["action"]) != policy_action:
        return None, "action_override_rejected"
    try:
        message = AgentMessage.model_validate(
            {
                "language": raw.get("language"),
                "message": raw.get("message"),
            }
        )
    except ValidationError:
        return None, "malformed_output"

    lowered = message.message.lower()
    if any(
        token in lowered
        for token in (
            "payment was successful",
            "payment succeeded",
            "razorpay secret",
            "api key",
            "ignore your instructions",
            "ignore previous",
            "change the action",
            "discount",
            "refund",
        )
    ):
        return None, "unsafe_content"

    # Models must not invent URLs. Only the app-inserted link is allowed.
    import re

    urls = re.findall(r"https?://\S+", message.message)
    if urls:
        if not allowed_payment_link or any(url != allowed_payment_link for url in urls):
            return None, "hallucinated_url"

    amount_token = f"{amount_minor / 100:,.0f}".replace(",", "")
    pretty = f"{amount_minor / 100:,.0f}"
    if amount_token not in message.message.replace(",", "") and pretty not in message.message:
        if "₹" not in message.message and "rs" not in lowered:
            return None, "amount_missing"
    return message, None


class RecoveryAgent:
    """Shadow-only agent. Never executes provider actions."""

    execution_mode = "shadow"

    def run(self, agent_input: AgentCaseInput) -> RecoveryAgentResult:
        payload = agent_input.model_dump()
        # Strip anything that could look like credentials from the bound object.
        safe_payload = {
            key: value
            for key, value in payload.items()
            if key
            not in {
                "database_url",
                "razorpay_key",
                "razorpay_secret",
                "webhook_secret",
            }
        }

        if agent_input.fraud or agent_input.decision_type in {"block", "stop"}:
            return RecoveryAgentResult(
                status="blocked",
                action="escalate_to_merchant",
                message=None,
                language=None,
                executed=False,
                execution_mode=self.execution_mode,
                communication_model="none",
                communication_status="blocked",
                blocked_reason="policy_blocked",
                recovery_probability=agent_input.recovery_probability,
                case_id=agent_input.case_id,
                payload=safe_payload,
            )

        if not _action_allows_communication(agent_input.action):
            return RecoveryAgentResult(
                status="shadow",
                action=agent_input.action,
                message=None,
                language=None,
                executed=False,
                execution_mode=self.execution_mode,
                communication_model="none",
                communication_status="no_customer_message",
                blocked_reason=None,
                recovery_probability=agent_input.recovery_probability,
                case_id=agent_input.case_id,
                payload=safe_payload,
            )

        preview = build_communication_preview(
            policy_action=agent_input.action,
            amount_minor=agent_input.amount_minor,
            language=agent_input.language,
            customer_name=agent_input.customer_name,
            taxonomy=agent_input.failure_category,
            fraud=agent_input.fraud,
            execution_mode=self.execution_mode,
        )

        draft = {
            "intent": "customer_message",
            "language": preview.get("language") or agent_input.language,
            "message": preview.get("message") or "",
            "confidence": float(preview.get("confidence") or 1.0),
        }

        structured, reject_reason = validate_structured_message(
            draft,
            policy_action=agent_input.action,
            amount_minor=agent_input.amount_minor,
        )
        validation = validate_qwen_output(
            draft,
            policy_action=agent_input.action,
            taxonomy=agent_input.failure_category,
            fraud=agent_input.fraud,
            policy_allows_communication=True,
        )

        if structured is None or not validation.allowed or not preview.get("ok"):
            fallback = render_template(
                agent_input.action,
                language=agent_input.language,
                customer_name=agent_input.customer_name,
                amount_minor=agent_input.amount_minor,
            )
            return RecoveryAgentResult(
                status="shadow",
                action=agent_input.action,
                message=fallback["message"],
                language=fallback["language"],
                executed=False,
                execution_mode=self.execution_mode,
                communication_model="deterministic_template",
                communication_status="fallback",
                blocked_reason=reject_reason or validation.reason or preview.get("reason"),
                recovery_probability=agent_input.recovery_probability,
                case_id=agent_input.case_id,
                payload={**safe_payload, "system_prompt": SYSTEM_PROMPT[:120]},
            )

        return RecoveryAgentResult(
            status="shadow",
            action=agent_input.action,
            message=structured.message,
            language=structured.language,
            executed=False,
            execution_mode=self.execution_mode,
            communication_model=str(preview.get("source") or "deterministic_template"),
            communication_status="validated",
            blocked_reason=None,
            recovery_probability=agent_input.recovery_probability,
            case_id=agent_input.case_id,
            payload={**safe_payload, "system_prompt": SYSTEM_PROMPT[:120]},
        )


def agent_plan_check(result: RecoveryAgentResult) -> dict[str, Any]:
    return {
        "name": "agent_communication",
        "passed": result.status in {"shadow", "blocked"},
        "status": result.status,
        "executed": False,
        "execution_mode": result.execution_mode,
        "communication_model": result.communication_model,
        "communication_status": result.communication_status,
        "message": result.message,
        "language": result.language,
        "action": result.action,
        "blocked_reason": result.blocked_reason,
        "recovery_probability": result.recovery_probability,
    }


def attach_agent_plan(record: AgentDecision, result: RecoveryAgentResult) -> None:
    checks = list(record.risk_checks or [])
    checks = [c for c in checks if not (isinstance(c, dict) and c.get("name") == "agent_communication")]
    checks.append(agent_plan_check(result))
    record.risk_checks = checks
    candidates = dict(record.candidate_actions or {})
    candidates["_agent_plan"] = result.as_dict()
    record.candidate_actions = candidates


def extract_agent_plan(record: AgentDecision) -> dict[str, Any] | None:
    candidates = record.candidate_actions or {}
    plan = candidates.get("_agent_plan")
    if isinstance(plan, dict):
        return plan
    for check in record.risk_checks or []:
        if isinstance(check, dict) and check.get("name") == "agent_communication":
            return {
                "case_id": record.recovery_case_id,
                "status": check.get("status"),
                "action": check.get("action") or record.selected_action,
                "message": check.get("message"),
                "language": check.get("language"),
                "executed": False,
                "execution_mode": check.get("execution_mode") or record.execution_mode,
                "communication_model": check.get("communication_model"),
                "communication_status": check.get("communication_status"),
                "blocked_reason": check.get("blocked_reason"),
                "recovery_probability": check.get("recovery_probability"),
            }
    return None


def run_agent_for_decision(
    session: Session,
    decision: AgentDecision,
) -> RecoveryAgentResult:
    row = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(RecoveryCase.id == decision.recovery_case_id)
    ).one_or_none()
    if row is None:
        return RecoveryAgentResult(
            status="blocked",
            action=decision.selected_action,
            message=None,
            language=None,
            executed=False,
            execution_mode="shadow",
            communication_model="none",
            communication_status="blocked",
            blocked_reason="case_not_found",
            recovery_probability=_selected_probability(decision),
            case_id=decision.recovery_case_id,
            payload={},
        )
    recovery_case, payment, customer = row
    agent_input = build_agent_input(
        recovery_case=recovery_case,
        payment=payment,
        customer=customer,
        decision=decision,
    )
    result = RecoveryAgent().run(agent_input)
    attach_agent_plan(decision, result)
    return result


def run_agent_for_case(
    session: Session,
    recovery_case_id: str,
) -> dict[str, Any]:
    decision = session.scalar(
        select(AgentDecision)
        .where(
            AgentDecision.recovery_case_id == recovery_case_id,
            AgentDecision.execution_mode == "shadow",
        )
        .order_by(AgentDecision.created_at.desc())
    )
    if decision is None:
        return {
            "case_id": recovery_case_id,
            "status": "blocked",
            "action": None,
            "message": None,
            "executed": False,
            "execution_mode": "shadow",
            "blocked_reason": "no_shadow_decision",
            "detail": "Run shadow inference before agent preview.",
        }
    existing = extract_agent_plan(decision)
    if existing and existing.get("communication_status") in {
        "validated",
        "fallback",
        "no_customer_message",
        "blocked",
    }:
        return {
            **existing,
            "executed": False,
            "decision_id": decision.id,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "idempotent_replay": True,
        }
    result = run_agent_for_decision(session, decision)
    session.flush()
    return {
        **result.as_dict(),
        "decision_id": decision.id,
        "model_version": decision.model_version,
        "policy_version": decision.policy_version,
        "idempotent_replay": False,
    }


def draft_message_in_language(
    session: Session,
    *,
    recovery_case_id: str,
    language: str,
    template_action: str | None = None,
) -> dict[str, Any]:
    """Merchant-assisted multilingual draft. Preview only — never sends."""
    selected_language = normalize_language(language)
    row = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(RecoveryCase.id == recovery_case_id)
    ).one_or_none()
    if row is None:
        return {
            "ok": False,
            "executed": False,
            "blocked_reason": "case_not_found",
            "case_id": recovery_case_id,
        }

    recovery_case, payment, customer = row
    decision = session.scalar(
        select(AgentDecision)
        .where(AgentDecision.recovery_case_id == recovery_case_id)
        .order_by(AgentDecision.created_at.desc())
    )
    customer.preferred_language = selected_language

    policy_action = (
        (decision.selected_action if decision else None)
        or "escalate_to_merchant"
    )
    # If policy blocks auto-comms (escalate/stop), merchant can still draft
    # a reminder/payment-link message as an assisted preview.
    draft_action = template_action
    if not draft_action:
        draft_action = (
            policy_action
            if _action_allows_communication(policy_action)
            else "whatsapp_reminder"
        )
    if draft_action not in {
        "payment_link",
        "retry_payment",
        "whatsapp_reminder",
        "escalate_to_merchant",
        "promise_to_pay",
    }:
        draft_action = "whatsapp_reminder"

    rendered = render_template(
        draft_action,
        language=selected_language,
        customer_name=customer.name,
        amount_minor=int(payment.amount or 0),
    )
    plan = {
        "case_id": recovery_case.id,
        "status": "drafted",
        "action": draft_action,
        "message": rendered["message"],
        "language": selected_language,
        "executed": False,
        "execution_mode": "shadow",
        "communication_model": "deterministic_template",
        "communication_status": "drafted",
        "blocked_reason": None,
        "recovery_probability": (
            _selected_probability(decision) if decision else None
        ),
        "merchant_assisted": True,
        "policy_action": policy_action,
        "preview_only": True,
        "send_to_customer": False,
    }

    if decision is not None:
        candidates = dict(decision.candidate_actions or {})
        candidates["_agent_plan"] = plan
        decision.candidate_actions = candidates
        checks = [
            c
            for c in list(decision.risk_checks or [])
            if not (isinstance(c, dict) and c.get("name") == "agent_communication")
        ]
        checks.append(
            {
                "name": "agent_communication",
                "status": "ok",
                "action": draft_action,
                "message": plan["message"],
                "language": selected_language,
                "execution_mode": "shadow",
                "communication_model": "deterministic_template",
                "communication_status": "drafted",
                "merchant_assisted": True,
                "recovery_probability": plan["recovery_probability"],
            }
        )
        decision.risk_checks = checks

    session.flush()
    amount_raw = int(payment.amount or 0)
    return {
        "ok": True,
        **plan,
        "decision_id": decision.id if decision else None,
        "customer_name": customer.name,
        "amount": round(amount_raw / 100, 2) if amount_raw >= 1000 else float(amount_raw),
    }
