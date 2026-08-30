from __future__ import annotations

"""
Phase 16 Qwen recovery agent.

Qwen communicates and explains. The deterministic policy remains financial
authority. This module fails closed and never executes provider actions.
"""

import json
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


from app.services.communication_templates import render_template


REPO_ROOT = Path(__file__).resolve().parents[3]
CONFIG_PATH = REPO_ROOT / "ml" / "config" / "qwen_agent.yaml"

INJECTION_PATTERNS = (
    r"ignore (all |previous )?policy",
    r"ignore (all |previous )?instructions",
    r"override (the )?policy",
    r"send another payment link",
    r"retry now",
    r"execute action",
)


class QwenAgentError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def load_qwen_agent_config() -> dict[str, Any]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    if config["authority"]["financial_decisions"]:
        raise QwenAgentError("Qwen cannot hold financial authority")
    if config["authority"]["tools_enabled"] or config["safety"]["qwen_tools_enabled"]:
        raise QwenAgentError("Qwen tools remain disabled")
    if config["safety"]["controlled_execution_authorized"]:
        raise QwenAgentError("Qwen config cannot authorize controlled execution")
    return config


@dataclass(frozen=True)
class QwenAgentResult:
    allowed: bool
    intent: str | None
    language: str | None
    message: str | None
    confidence: float | None
    reason: str
    raw: dict[str, Any] | None = None


def _detect_injection(text: str) -> bool:
    lowered = text.lower()
    return any(re.search(pattern, lowered) for pattern in INJECTION_PATTERNS)


def validate_qwen_output(
    payload: dict[str, Any],
    *,
    policy_action: str | None,
    taxonomy: str | None,
    fraud: bool,
    policy_allows_communication: bool,
) -> QwenAgentResult:
    config = load_qwen_agent_config()
    required = config["output_schema"]["required"]
    if any(key not in payload for key in required):
        return QwenAgentResult(
            allowed=False,
            intent=None,
            language=None,
            message=None,
            confidence=None,
            reason="malformed_output",
            raw=payload,
        )

    intent = str(payload["intent"])
    language = str(payload["language"]).lower()
    message = str(payload["message"])
    try:
        confidence = float(payload["confidence"])
    except (TypeError, ValueError):
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=None,
            reason="malformed_output",
            raw=payload,
        )

    if intent in config["forbidden_intents"]:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="unsupported_action",
            raw=payload,
        )
    if intent not in config["allowed_intents"]:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="unsupported_action",
            raw=payload,
        )
    if language not in config["output_schema"]["languages"]:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="unsupported_language",
            raw=payload,
        )
    if not 0 <= confidence <= 1:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="malformed_output",
            raw=payload,
        )
    if fraud and config["fail_closed"]["fraud_case"]:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="fraud_case",
            raw=payload,
        )
    if (
        taxonomy in {None, "", "unknown"}
        and config["fail_closed"]["unknown_taxonomy"]
        and intent != "diagnosis_explanation"
    ):
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="unknown_taxonomy",
            raw=payload,
        )
    if not policy_allows_communication:
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="policy_blocked",
            raw=payload,
        )
    if _detect_injection(message) or _detect_injection(json.dumps(payload)):
        return QwenAgentResult(
            allowed=False,
            intent=intent,
            language=language,
            message=None,
            confidence=confidence,
            reason="prompt_injection",
            raw=payload,
        )
    if policy_action is None and intent == "customer_message":
        # Communication may still explain a block, but cannot imply an action.
        if any(
            token in message.lower()
            for token in ("payment link", "retry now", "whatsapp")
        ):
            return QwenAgentResult(
                allowed=False,
                intent=intent,
                language=language,
                message=None,
                confidence=confidence,
                reason="hallucinated_payment_status",
                raw=payload,
            )

    return QwenAgentResult(
        allowed=True,
        intent=intent,
        language=language,
        message=message,
        confidence=confidence,
        reason="validated",
        raw=payload,
    )


def draft_customer_message(
    *,
    amount_minor: int,
    language: str = "english",
    customer_name: str | None = None,
    action: str = "payment_link",
) -> dict[str, Any]:
    """Deterministic template draft used until a real Qwen backend is wired."""
    return render_template(
        action,
        language=language,
        customer_name=customer_name,
        amount_minor=amount_minor,
    )
