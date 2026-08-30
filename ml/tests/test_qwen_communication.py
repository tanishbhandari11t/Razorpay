from __future__ import annotations

from ml.src.communication_policy import communication_allowed
from ml.src.communication_validator import validate_communication_payload
from ml.src.qwen_client import generate_with_qwen, qwen_runtime_status


def test_qwen_client_stays_disabled() -> None:
    status = qwen_runtime_status()
    assert status["tools_enabled"] is False
    assert status["financial_authority"] is False
    assert status["send_to_customer"] is False
    result = generate_with_qwen(
        action="payment_link",
        amount_minor=249900,
        language="hinglish",
    )
    assert result["ok"] is False


def test_communication_policy_preview_only() -> None:
    decision = communication_allowed(
        policy_action="payment_link",
        execution_mode="shadow",
        fraud=False,
        taxonomy="customer",
    )
    assert decision["allowed"] is True
    assert decision["send_to_customer"] is False


def test_communication_validator_requires_schema() -> None:
    bad = validate_communication_payload({"intent": "customer_message"})
    assert bad["valid"] is False
    good = validate_communication_payload(
        {
            "intent": "customer_message",
            "language": "english",
            "message": "hello",
            "confidence": 0.9,
        }
    )
    assert good["valid"] is True
    assert good["send_to_customer"] is False
