from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
import yaml

from ml.src.action_executor import ExecutionStatus, execute_decision
from ml.src.failure_classifier import FailureClass, classify_failure
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    DecisionType,
    decide_recovery_action,
)
from ml.src.policies.decision_margin import apply_decision_margin
from ml.src.policies.stopping_rules import (
    InterventionAttempt,
    RecoveryPolicyContext,
)
from ml.src.recovery_state_machine import RecoveryState, RecoveryStateMachine


REPO_ROOT = Path(__file__).resolve().parents[2]


def context(
    *,
    amount: float = 1000,
    reason: str = "temporary_failure",
    fraud: int = 0,
    status: str = "failed",
    attempts: list[InterventionAttempt] | None = None,
) -> RecoveryPolicyContext:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    return RecoveryPolicyContext(
        case_id="CASE-1",
        payment_id="PAY-1",
        amount_inr=amount,
        payment_status=status,
        failure_reason=reason,
        fraud_flag=fraud,
        case_created_at=now - timedelta(hours=1),
        now=now,
        customer_contact_available=True,
        customer_opted_out=False,
        valid_payment_context=True,
        attempts=attempts or [],
    )


def probabilities() -> dict[str, float]:
    return {
        "retry_payment": 0.60,
        "payment_link": 0.65,
        "whatsapp_reminder": 0.50,
        "escalate_to_merchant": 0.30,
    }


def supported() -> dict[str, CandidateSupport]:
    return {
        action: CandidateSupport(True, 100, 50)
        for action in probabilities()
    }


@pytest.mark.parametrize(
    ("reason", "expected"),
    [
        ("gateway_timeout", FailureClass.TECHNICAL_RETRYABLE),
        ("insufficient_funds", FailureClass.INSUFFICIENT_FUNDS),
        ("otp_failed", FailureClass.AUTHENTICATION_REQUIRED),
        ("merchant_not_enabled", FailureClass.MERCHANT_CONFIGURATION),
        ("card_declined", FailureClass.CUSTOMER_DECLINE),
        ("unmapped_reason", FailureClass.UNKNOWN),
    ],
)
def test_failure_classifier_taxonomy(reason, expected) -> None:
    assert classify_failure(reason).failure_class == expected


def test_fraud_never_selects_automated_action() -> None:
    decision = decide_recovery_action(
        context(fraud=1),
        probabilities(),
        supported(),
    )
    assert decision.decision_type == DecisionType.FALLBACK
    assert decision.selected_action == "escalate_to_merchant"
    assert "fraud_requires_manual_escalation" in decision.reasons


def test_recovered_payment_stops_without_action() -> None:
    decision = decide_recovery_action(
        context(status="captured"),
        probabilities(),
        supported(),
    )
    assert decision.decision_type == DecisionType.STOP
    assert decision.selected_action is None
    assert decision.reasons == ("payment_already_recovered",)


def test_intervention_budget_stops_case() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    attempts = [
        InterventionAttempt("retry_payment", "failed", now - timedelta(hours=3)),
        InterventionAttempt("payment_link", "failed", now - timedelta(hours=2)),
        InterventionAttempt("whatsapp_reminder", "failed", now - timedelta(hours=1)),
    ]
    decision = decide_recovery_action(
        context(attempts=attempts),
        probabilities(),
        supported(),
    )
    assert decision.decision_type == DecisionType.STOP
    assert "maximum_intervention_budget_reached" in decision.reasons


def test_action_cooldown_blocks_immediate_repeat() -> None:
    now = datetime(2024, 1, 2, tzinfo=UTC)
    attempts = [
        InterventionAttempt(
            "retry_payment",
            "failed",
            now - timedelta(minutes=30),
        )
    ]
    decision = decide_recovery_action(
        context(amount=1000, attempts=attempts),
        probabilities(),
        supported(),
    )
    assert not decision.candidates["retry_payment"].eligible
    assert any(
        reason.startswith("cooldown_until:")
        for reason in decision.candidates[
            "retry_payment"
        ].eligibility_reasons
    )
    assert decision.selected_action == "payment_link"


def test_customer_opt_out_stops_contact() -> None:
    value = context()
    value.customer_opted_out = True
    decision = decide_recovery_action(value, probabilities(), supported())
    assert decision.decision_type == DecisionType.STOP
    assert decision.reasons == ("customer_opted_out",)


def test_expired_recovery_window_stops_case() -> None:
    value = context()
    value.case_created_at = value.now - timedelta(hours=73)
    decision = decide_recovery_action(value, probabilities(), supported())
    assert decision.decision_type == DecisionType.STOP
    assert decision.reasons == ("recovery_window_expired",)


def test_expected_value_changes_action_with_payment_amount() -> None:
    small = decide_recovery_action(
        context(amount=100),
        probabilities(),
        supported(),
    )
    large = decide_recovery_action(
        context(amount=1000),
        probabilities(),
        supported(),
    )
    assert small.selected_action == "retry_payment"
    assert large.selected_action == "payment_link"
    assert (
        large.expected_values["payment_link"]
        > large.expected_values["retry_payment"]
    )


def test_minimum_expected_value_stops_low_value_case() -> None:
    decision = decide_recovery_action(
        context(amount=20),
        probabilities(),
        supported(),
    )
    assert decision.decision_type == DecisionType.STOP
    assert decision.selected_action is None
    assert "no_eligible_positive_value_action" in decision.reasons


def test_phase13_decision_margin_falls_back_when_value_gain_is_weak() -> None:
    decision = decide_recovery_action(
        context(amount=1000),
        probabilities(),
        supported(),
    )
    gated = apply_decision_margin(
        decision,
        fallback_action="retry_payment",
        threshold_inr=50,
    )
    assert decision.selected_action == "payment_link"
    assert gated.selected_action == "retry_payment"
    assert gated.fallback_triggered is True
    assert gated.decision_margin_inr == pytest.approx(42.0)


def test_phase13_decision_margin_preserves_strong_selection() -> None:
    decision = decide_recovery_action(
        context(amount=1000),
        probabilities(),
        supported(),
    )
    gated = apply_decision_margin(
        decision,
        fallback_action="retry_payment",
        threshold_inr=10,
    )
    assert gated.selected_action == "payment_link"
    assert gated.fallback_triggered is False


def test_unsupported_highest_probability_is_not_selected() -> None:
    support = supported()
    support["payment_link"] = CandidateSupport(False, 2, 2)
    decision = decide_recovery_action(
        context(amount=1000),
        probabilities(),
        support,
    )
    assert decision.selected_action == "retry_payment"
    assert not decision.candidates["payment_link"].eligible


def test_state_machine_rejects_unbounded_transition() -> None:
    machine = RecoveryStateMachine()
    with pytest.raises(ValueError, match="Invalid recovery transition"):
        machine.transition(RecoveryState.ACTION_EXECUTED)


def test_dry_run_executor_never_calls_provider() -> None:
    decision = decide_recovery_action(
        context(amount=1000),
        probabilities(),
        supported(),
    )
    result = execute_decision(decision)
    assert result.status == ExecutionStatus.WOULD_EXECUTE
    assert result.message.startswith("WOULD_EXECUTE:")
    assert result.dry_run is True


def test_phase7_manifest_remains_frozen() -> None:
    manifest = yaml.safe_load(
        (
            REPO_ROOT / "ml" / "config" / "policy_v2_manifest.yaml"
        ).read_text()
    )
    assert manifest["status"]["production_approved"] is False
    assert (
        manifest["benchmarks"]["support_safe_v2_doubly_robust_estimate"]
        == 0.62439680
    )
