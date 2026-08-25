from __future__ import annotations

from copy import deepcopy

import numpy as np
import pandas as pd
import pytest

from ml.src.create_logging_policy import create_logging_dataset
from ml.src.policies.behavior_policy import (
    base_policy_action,
    choose_action,
    load_policy_config,
)
from ml.src.simulate_recovery import load_config, simulate_outcomes


def feature(index: int, *, fraud: bool = False) -> dict[str, object]:
    timestamp = pd.Timestamp("2024-01-01") + pd.Timedelta(days=index)
    return {
        "transaction_id": f"T{index:04d}",
        "customer_id": f"C{index % 8:03d}",
        "prediction_time": timestamp,
        "amount_inr": 1000 + index,
        "transaction_type": "P2M",
        "merchant_category": "Grocery",
        "device_type": "Android",
        "network_type": "4G",
        "fraud_flag": int(fraud),
        "hour_of_day": 12,
        "day_of_week": timestamp.day_name(),
        "is_weekend": int(timestamp.dayofweek >= 5),
        "sender_age_group": "26-35",
        "sender_state": "Karnataka",
        "sender_bank": "HDFC",
        "has_prior_history": 1,
        "has_previous_success": 1,
        "has_previous_failure": 1,
        "previous_transaction_count": 20,
        "previous_success_count": 18,
        "previous_failure_count": 2,
        "historical_success_rate": 0.9,
        "historical_failure_rate": 0.1,
        "previous_failure_streak": 0,
        "transactions_last_7d": 2,
        "successes_last_7d": 2,
        "failures_last_7d": 0,
        "amount_last_7d": 1800.0,
        "transactions_last_30d": 5,
        "successes_last_30d": 4,
        "failures_last_30d": 1,
        "amount_last_30d": 4500.0,
        "days_since_previous_transaction": 2.0,
        "days_since_previous_success": 3.0,
        "days_since_previous_failure": 20.0,
        "previous_avg_amount": 900.0,
        "previous_median_amount": 850.0,
        "previous_max_amount": 2000.0,
        "amount_vs_previous_avg": 1.2,
        "current_amount_percentile": 0.7,
        "same_transaction_type_previous_count": 15,
        "same_merchant_category_previous_count": 8,
        "same_merchant_category_previous_rate": 0.4,
        "customer_primary_device_before_failure": "Android",
        "device_matches_primary": 1,
        "customer_primary_network_before_failure": "4G",
        "network_matches_primary": 1,
        "previous_fraud_count": 0,
        "historical_fraud_rate": 0.0,
        "historical_weekend_ratio": 0.25,
        "same_day_of_week_previous_rate": 0.15,
        "usual_hour_previous_rate": 0.5,
    }


def fixture_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    features = pd.DataFrame(
        [feature(index, fraud=index == 7) for index in range(40)]
    )
    outcomes = simulate_outcomes(features, load_config())
    return features, outcomes


def test_base_policy_rules_are_ordered() -> None:
    config = load_policy_config()
    row = feature(1)
    assert base_policy_action(row, config) == "retry_payment"

    row["days_since_previous_success"] = 20
    row["amount_vs_previous_avg"] = 2
    assert base_policy_action(row, config) == "payment_link"

    row["historical_success_rate"] = 0.6
    assert base_policy_action(row, config) == "whatsapp_reminder"

    row["transactions_last_30d"] = 0
    assert base_policy_action(row, config) == "escalate_to_merchant"


def test_fraud_always_produces_no_action() -> None:
    decision = choose_action(feature(7, fraud=True), load_policy_config())
    assert decision.chosen_intervention == "no_action"
    assert decision.policy_probability == 1
    assert decision.policy_type == "policy_blocked"


def test_behavior_and_exploration_propensities_are_exact() -> None:
    config = load_policy_config()
    decisions = [choose_action(feature(index), config) for index in range(500)]
    behavior = [value for value in decisions if value.policy_type == "behavior"]
    exploration = [
        value for value in decisions if value.policy_type == "exploration"
    ]

    assert behavior
    assert exploration
    assert all(value.policy_probability == pytest.approx(0.9) for value in behavior)
    assert all(
        value.policy_probability == pytest.approx(0.1 / 3)
        for value in exploration
    )
    assert all(
        value.chosen_intervention != value.base_policy_intervention
        for value in exploration
    )


def test_logging_dataset_has_one_row_per_payment() -> None:
    features, outcomes = fixture_data()
    dataset, _ = create_logging_dataset(
        features,
        outcomes,
        load_policy_config(),
    )

    assert len(dataset) == len(features)
    assert dataset["payment_id"].is_unique
    assert set(dataset["payment_id"]) == set(features["transaction_id"])


def test_selected_outcome_matches_potential_outcome() -> None:
    features, outcomes = fixture_data()
    dataset, _ = create_logging_dataset(
        features,
        outcomes,
        load_policy_config(),
    )
    observed = dataset.loc[dataset["chosen_intervention"].ne("no_action")]
    selected = outcomes.merge(
        observed[["payment_id", "chosen_intervention"]],
        left_on=["payment_id", "intervention"],
        right_on=["payment_id", "chosen_intervention"],
        validate="one_to_one",
    ).set_index("payment_id")
    logged = observed.set_index("payment_id")

    np.testing.assert_array_equal(
        logged.loc[selected.index, "recovered"],
        selected["recovered"],
    )
    np.testing.assert_array_equal(
        logged.loc[selected.index, "amount_recovered"],
        selected["amount_recovered"],
    )


def test_simulator_answers_and_counterfactuals_are_not_exposed() -> None:
    features, outcomes = fixture_data()
    dataset, _ = create_logging_dataset(
        features,
        outcomes,
        load_policy_config(),
    )

    forbidden = {
        "simulated_recovery_probability",
        "synthetic_failure_scenario",
        "simulation_version",
    }
    assert forbidden.isdisjoint(dataset.columns)
    assert not any(column.startswith("counterfactual_") for column in dataset)


def test_temporal_splits_are_strictly_ordered() -> None:
    features, outcomes = fixture_data()
    dataset, _ = create_logging_dataset(
        features,
        outcomes,
        load_policy_config(),
    )
    timestamps = pd.to_datetime(dataset["prediction_time"])

    train = timestamps.loc[dataset["split"].eq("train")]
    validation = timestamps.loc[dataset["split"].eq("validation")]
    test = timestamps.loc[dataset["split"].eq("test")]
    assert train.max() < validation.min()
    assert validation.max() < test.min()


def test_logging_policy_is_deterministic_under_reordering() -> None:
    features, outcomes = fixture_data()
    config = load_policy_config()
    first, first_cutoffs = create_logging_dataset(features, outcomes, config)
    second, second_cutoffs = create_logging_dataset(
        features.sample(frac=1, random_state=11).reset_index(drop=True),
        outcomes.sample(frac=1, random_state=12).reset_index(drop=True),
        config,
    )

    pd.testing.assert_frame_equal(first, second)
    assert first_cutoffs == second_cutoffs


def test_invalid_split_configuration_is_rejected(tmp_path) -> None:
    config = deepcopy(load_policy_config())
    config["temporal_split"]["test_fraction"] = 0.5
    path = tmp_path / "invalid.yaml"
    import yaml

    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    with pytest.raises(ValueError, match="sum to 1"):
        load_policy_config(path)
