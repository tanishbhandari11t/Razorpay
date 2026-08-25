from __future__ import annotations

from copy import deepcopy

import pandas as pd
import pytest

from ml.src.simulate_recovery import (
    EXPECTED_INTERVENTIONS,
    load_config,
    simulate_outcomes,
)


def feature(
    transaction_id: str,
    customer_id: str,
    *,
    fraud_flag: int = 0,
    amount_ratio: float = 1.2,
    success_rate: float = 0.9,
) -> dict[str, object]:
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "prediction_time": "2024-08-01 12:00:00",
        "amount_inr": 2499,
        "fraud_flag": fraud_flag,
        "has_prior_history": 1,
        "has_previous_success": 1,
        "historical_success_rate": success_rate,
        "previous_transaction_count": 20,
        "previous_failure_streak": 0,
        "failures_last_7d": 0,
        "failures_last_30d": 1,
        "transactions_last_30d": 5,
        "days_since_previous_success": 3.0,
        "amount_vs_previous_avg": amount_ratio,
        "device_matches_primary": 1,
        "network_matches_primary": 1,
    }


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            feature("T001", "C001"),
            feature("T002", "C002", fraud_flag=1),
            feature("T003", "C003", amount_ratio=192.6267, success_rate=0.4),
        ]
    )


def test_every_payment_gets_four_potential_outcomes() -> None:
    outcomes = simulate_outcomes(feature_frame(), load_config())

    assert len(outcomes) == len(feature_frame()) * 4
    assert not outcomes.duplicated(["payment_id", "intervention"]).any()
    assert set(outcomes["intervention"]) == set(EXPECTED_INTERVENTIONS)
    assert outcomes.groupby("payment_id")["intervention"].nunique().eq(4).all()


def test_simulation_is_deterministic_under_input_reordering() -> None:
    features = feature_frame()
    config = load_config()
    first = simulate_outcomes(features, config)
    second = simulate_outcomes(
        features.sample(frac=1, random_state=99).reset_index(drop=True),
        config,
    )

    pd.testing.assert_frame_equal(first, second)


def test_fraud_policy_blocks_automated_actions_only() -> None:
    outcomes = simulate_outcomes(feature_frame(), load_config())
    fraud = outcomes.loc[outcomes["payment_id"].eq("T002")].set_index(
        "intervention"
    )

    for intervention in EXPECTED_INTERVENTIONS[:3]:
        assert fraud.loc[intervention, "policy_allowed"] == 0
        assert fraud.loc[intervention, "simulated_recovery_probability"] == 0
        assert fraud.loc[intervention, "recovered"] == 0
    assert fraud.loc["escalate_to_merchant", "policy_allowed"] == 1


def test_extreme_amount_ratio_is_robustly_bounded() -> None:
    outcomes = simulate_outcomes(feature_frame(), load_config())
    extreme = outcomes.loc[outcomes["payment_id"].eq("T003")]

    assert extreme["simulated_recovery_probability"].between(0, 0.95).all()
    assert not extreme["simulated_recovery_probability"].isna().any()


def test_recovered_money_and_time_are_coherent() -> None:
    outcomes = simulate_outcomes(feature_frame(), load_config())
    recovered = outcomes["recovered"].eq(1)

    assert outcomes.loc[~recovered, "amount_recovered"].eq(0).all()
    assert outcomes.loc[~recovered, "time_to_recovery_hours"].eq(0).all()
    assert outcomes.loc[recovered, "amount_recovered"].eq(
        outcomes.loc[recovered, "amount_inr"]
    ).all()
    assert outcomes.loc[recovered, "time_to_recovery_hours"].gt(0).all()


def test_hidden_traits_are_not_exposed_as_model_columns() -> None:
    outcomes = simulate_outcomes(feature_frame(), load_config())
    forbidden = {
        "customer_responsiveness",
        "customer_intervention_preference",
        "payment_specific_shock",
    }
    assert forbidden.isdisjoint(outcomes.columns)


def test_invalid_scenario_probabilities_are_rejected(tmp_path) -> None:
    config = deepcopy(load_config())
    config["scenario_probabilities"]["temporary_failure"] = 0.99
    path = tmp_path / "invalid.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        load_config(path)
