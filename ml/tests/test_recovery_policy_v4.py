from __future__ import annotations

from datetime import UTC, datetime

from ml.src.policies.recovery_policy import CandidateSupport
from ml.src.policies.recovery_policy_v4 import (
    decide_recovery_action_v4,
    load_policy_v4,
)
from ml.src.policies.stopping_rules import RecoveryPolicyContext


ACTIONS = (
    "retry_payment",
    "payment_link",
    "whatsapp_reminder",
    "escalate_to_merchant",
)


def _context(*, amount: float, reason: str, fraud: int = 0) -> RecoveryPolicyContext:
    now = datetime(2026, 8, 25, tzinfo=UTC)
    return RecoveryPolicyContext(
        case_id="case-v4",
        payment_id="payment-v4",
        amount_inr=amount,
        payment_status="failed",
        failure_reason=reason,
        fraud_flag=fraud,
        case_created_at=now,
        now=now,
        customer_contact_available=True,
        valid_payment_context=True,
    )


def _support() -> dict[str, CandidateSupport]:
    return {
        action: CandidateSupport(
            supported=True,
            action_count=100,
            effective_sample_size=50,
        )
        for action in ACTIONS
    }


def test_v4_uses_no_action_when_incremental_value_is_too_small() -> None:
    decision = decide_recovery_action_v4(
        _context(amount=10, reason="gateway_timeout"),
        {
            "retry_payment": 0.6,
            "payment_link": 0.7,
            "whatsapp_reminder": 0.65,
            "escalate_to_merchant": 0.4,
            "no_action": 0.0,
        },
        _support(),
    )
    assert decision.selected_action == "no_action"
    assert decision.execution_authorized is False


def test_v4_preserves_supported_v3_action_after_economic_gate() -> None:
    decision = decide_recovery_action_v4(
        _context(amount=1000, reason="gateway_timeout"),
        {
            "retry_payment": 0.9,
            "payment_link": 0.2,
            "whatsapp_reminder": 0.2,
            "escalate_to_merchant": 0.1,
            "no_action": 0.0,
        },
        _support(),
    )
    assert decision.selected_action == "retry_payment"
    assert decision.candidates["retry_payment"].incremental_net_value_inr > 0


def test_v4_preserves_manual_fraud_escalation() -> None:
    decision = decide_recovery_action_v4(
        _context(amount=1000, reason="suspected_fraud", fraud=1),
        {
            "retry_payment": 0.9,
            "payment_link": 0.9,
            "whatsapp_reminder": 0.9,
            "escalate_to_merchant": 0.1,
            "no_action": 0.0,
        },
        _support(),
    )
    assert decision.selected_action == "escalate_to_merchant"
    assert decision.decision_type == "fallback"


def test_v4_frozen_manifests_are_unchanged() -> None:
    config = load_policy_v4()
    assert config["base_policy_version"] == "recovery_policy_v3"
    assert config["safety"]["execution_authorized"] is False
