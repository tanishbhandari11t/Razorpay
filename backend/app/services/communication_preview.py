from __future__ import annotations

"""
Fail-closed communication preview.

Policy already chose the action. This layer only drafts a message preview.
Nothing is sent to the customer and no provider tools are called.
"""

from typing import Any

from app.services.communication_templates import render_template
from app.services.qwen_agent import draft_customer_message, validate_qwen_output


def build_communication_preview(
    *,
    policy_action: str,
    amount_minor: int,
    language: str = "english",
    customer_name: str | None = None,
    taxonomy: str | None = None,
    fraud: bool = False,
    execution_mode: str = "shadow",
) -> dict[str, Any]:
    if fraud:
        return {
            "ok": False,
            "executed": False,
            "send_to_customer": False,
            "source": "blocked",
            "reason": "fraud_case",
            "message": None,
            "execution_mode": execution_mode,
        }

    draft = draft_customer_message(
        amount_minor=amount_minor,
        language=language,
        customer_name=customer_name,
        action=policy_action,
    )
    validation = validate_qwen_output(
        draft,
        policy_action=policy_action,
        taxonomy=taxonomy,
        fraud=fraud,
        policy_allows_communication=True,
    )
    if not validation.allowed:
        fallback = render_template(
            "escalate_to_merchant",
            language=language,
            customer_name=customer_name,
            amount_minor=amount_minor,
        )
        return {
            "ok": False,
            "executed": False,
            "send_to_customer": False,
            "source": "template_fallback",
            "reason": validation.reason,
            "message": fallback["message"],
            "language": fallback["language"],
            "intent": fallback["intent"],
            "confidence": fallback["confidence"],
            "policy_action": policy_action,
            "execution_mode": execution_mode,
            "qwen_tools_enabled": False,
            "financial_authority": False,
        }

    return {
        "ok": True,
        "executed": False,
        "send_to_customer": False,
        "source": draft.get("source", "deterministic_template"),
        "reason": "preview_only",
        "message": validation.message,
        "language": validation.language,
        "intent": validation.intent,
        "confidence": validation.confidence,
        "policy_action": policy_action,
        "execution_mode": execution_mode,
        "qwen_tools_enabled": False,
        "financial_authority": False,
        "note": (
            "Communication preview only. Policy decided the action; "
            "Qwen/tools cannot execute recovery."
        ),
    }
