from __future__ import annotations

"""Policy boundary for communication: Qwen may speak only about allowed actions."""

from typing import Any


BLOCKED_ACTIONS = {"retry_payment", "payment_link", "whatsapp_reminder"}


def communication_allowed(
    *,
    policy_action: str | None,
    execution_mode: str,
    fraud: bool,
    taxonomy: str | None,
) -> dict[str, Any]:
    if fraud:
        return {
            "allowed": False,
            "reason": "fraud_case",
            "preview_only": True,
            "send_to_customer": False,
        }
    if taxonomy in {None, "", "unknown"} and policy_action in BLOCKED_ACTIONS:
        return {
            "allowed": False,
            "reason": "unknown_taxonomy",
            "preview_only": True,
            "send_to_customer": False,
        }
    if execution_mode not in {"shadow", "controlled", "dry_run"}:
        return {
            "allowed": False,
            "reason": "invalid_execution_mode",
            "preview_only": True,
            "send_to_customer": False,
        }
    # Shadow previews are allowed for diagnosis/customer_message templates.
    return {
        "allowed": True,
        "reason": "preview_allowed",
        "preview_only": True,
        "send_to_customer": False,
        "action": policy_action,
        "execution_mode": execution_mode,
    }
