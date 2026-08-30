from __future__ import annotations

import pandas as pd
import pytest

from ml.src.analyze_support import verify_frozen_v1
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    context_buckets,
    load_support_policy_config,
    select_support_safe_action,
)


def row(*, fraud: int = 0) -> dict[str, float | int]:
    return {
        "historical_success_rate": 0.9,
        "has_prior_history": 1,
        "amount_inr": 500,
        "failures_last_30d": 0,
        "fraud_flag": fraud,
    }


def support_index(
    supported_actions: set[str],
    config: dict,
) -> SupportIndex:
    history, amount, failures = context_buckets(row(), config)
    table = pd.DataFrame(
        [
            {
                "history_bucket": history,
                "amount_bucket": amount,
                "failure_bucket": failures,
                "action": action,
                "action_count": 30 if action in supported_actions else 2,
                "effective_sample_size": (
                    15.0 if action in supported_actions else 2.0
                ),
            }
            for action in config["interventions"]
        ]
    )
    return SupportIndex(table, config)


def probabilities() -> dict[str, float]:
    return {
        "retry_payment": 0.70,
        "payment_link": 0.80,
        "whatsapp_reminder": 0.60,
        "escalate_to_merchant": 0.30,
    }


def test_unsupported_best_action_cannot_be_model_selected() -> None:
    config = load_support_policy_config()
    index = support_index(
        {"retry_payment", "whatsapp_reminder"},
        config,
    )
    decision = select_support_safe_action(
        row(),
        probabilities(),
        index,
        config,
    )

    assert decision.raw_best_action == "payment_link"
    assert decision.selected_action == "retry_payment"
    assert decision.selected_action_supported


def test_fraud_blocks_all_automated_actions() -> None:
    config = load_support_policy_config()
    index = support_index(set(config["interventions"]), config)
    decision = select_support_safe_action(
        row(fraud=1),
        probabilities(),
        index,
        config,
    )

    assert decision.selected_action == "no_action"
    assert decision.decision_reason == "fraud_policy_block"


def test_no_supported_candidate_activates_manual_fallback() -> None:
    config = load_support_policy_config()
    decision = select_support_safe_action(
        row(),
        probabilities(),
        support_index(set(), config),
        config,
    )

    assert decision.selected_action == "escalate_to_merchant"
    assert decision.fallback_used
    assert decision.decision_reason == "no_supported_candidate_safety_fallback"
    assert not decision.selected_action_supported


def test_small_decision_margin_uses_supported_retry_fallback() -> None:
    config = load_support_policy_config()
    candidate_probabilities = probabilities()
    candidate_probabilities["payment_link"] = 0.62
    candidate_probabilities["retry_payment"] = 0.61
    decision = select_support_safe_action(
        row(),
        candidate_probabilities,
        support_index(set(config["interventions"]), config),
        config,
    )

    assert decision.decision_margin == pytest.approx(0.01)
    assert decision.selected_action == "retry_payment"
    assert decision.fallback_used
    assert decision.decision_reason == "insufficient_probability_margin"


def test_invalid_candidate_probability_is_rejected() -> None:
    config = load_support_policy_config()
    candidate_probabilities = probabilities()
    candidate_probabilities["payment_link"] = 1.01
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        select_support_safe_action(
            row(),
            candidate_probabilities,
            support_index(set(config["interventions"]), config),
            config,
        )


def test_frozen_v1_artifacts_are_unchanged() -> None:
    config = load_support_policy_config()
    verified = verify_frozen_v1(config)
    assert verified["model_sha256"] == config["frozen_artifacts"]["model_sha256"]
