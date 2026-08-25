from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.policies.behavior_policy import (
    DEFAULT_CONFIG_PATH,
    choose_action,
    load_policy_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FEATURES_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "failed_payment_features.csv"
)
DEFAULT_OUTCOMES_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "intervention_outcomes.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "logging_policy_dataset.csv"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "logging_policy_summary.json"
)

REQUIRED_OUTCOME_COLUMNS = {
    "payment_id",
    "customer_id",
    "intervention",
    "policy_allowed",
    "recovered",
    "amount_inr",
    "amount_recovered",
    "intervention_cost",
    "net_recovered",
    "time_to_recovery_hours",
}
FORBIDDEN_LOGGING_COLUMNS = {
    "simulated_recovery_probability",
    "synthetic_failure_scenario",
    "simulation_version",
}


def _validate_inputs(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
) -> None:
    required_features = {
        "transaction_id",
        "customer_id",
        "prediction_time",
        "fraud_flag",
        "has_prior_history",
        "has_previous_success",
        "historical_success_rate",
        "days_since_previous_success",
        "amount_vs_previous_avg",
        "transactions_last_30d",
    }
    missing_features = required_features - set(features.columns)
    if missing_features:
        raise ValueError(f"Missing policy features: {sorted(missing_features)}")
    missing_outcomes = REQUIRED_OUTCOME_COLUMNS - set(outcomes.columns)
    if missing_outcomes:
        raise ValueError(f"Missing potential outcomes: {sorted(missing_outcomes)}")
    if features["transaction_id"].duplicated().any():
        raise ValueError("Duplicate failed-payment features found")
    if outcomes.duplicated(["payment_id", "intervention"]).any():
        raise ValueError("Duplicate potential outcomes found")
    if features.isna().any().any() or outcomes.isna().any().any():
        raise ValueError("Missing values found in logging-policy inputs")


def _assign_temporal_splits(
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    ordered = dataset.sort_values(
        ["prediction_time", "payment_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    timestamps = pd.to_datetime(ordered["prediction_time"], errors="raise")
    fractions = config["temporal_split"]
    validation_position = int(len(ordered) * float(fractions["train_fraction"]))
    test_position = int(
        len(ordered)
        * (
            float(fractions["train_fraction"])
            + float(fractions["validation_fraction"])
        )
    )
    if not 0 < validation_position < test_position < len(ordered):
        raise ValueError("Temporal split produced an empty partition")

    validation_start = timestamps.iloc[validation_position]
    test_start = timestamps.iloc[test_position]
    if validation_start >= test_start:
        raise ValueError("Temporal split cutoffs are not strictly ordered")

    ordered["split"] = np.select(
        [
            timestamps.lt(validation_start),
            timestamps.lt(test_start),
        ],
        ["train", "validation"],
        default="test",
    )
    if set(ordered["split"]) != {"train", "validation", "test"}:
        raise ValueError("Temporal split produced an empty partition")
    return ordered, {
        "validation_start": validation_start.isoformat(),
        "test_start": test_start.isoformat(),
    }


def create_logging_dataset(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, str]]:
    _validate_inputs(features, outcomes)
    potential_lookup = outcomes.set_index(["payment_id", "intervention"])
    records: list[dict[str, Any]] = []

    ordered_features = features.copy()
    ordered_features["prediction_time"] = pd.to_datetime(
        ordered_features["prediction_time"],
        errors="raise",
    )
    ordered_features = ordered_features.sort_values(
        ["prediction_time", "transaction_id"],
        kind="mergesort",
    )

    for _, feature in ordered_features.iterrows():
        decision = choose_action(feature, config)
        payment_id = str(feature["transaction_id"])
        record = feature.to_dict()
        record["payment_id"] = record.pop("transaction_id")
        record.update(
            {
                "chosen_intervention": decision.chosen_intervention,
                "base_policy_intervention": decision.base_policy_intervention,
                "policy_probability": decision.policy_probability,
                "policy_type": decision.policy_type,
                "logging_policy_version": int(config["version"]),
            }
        )

        if decision.chosen_intervention == config["fraud_action"]:
            observed = {
                "recovered": 0,
                "amount_recovered": 0,
                "intervention_cost": 0.0,
                "net_recovered": 0.0,
                "time_to_recovery_hours": 0.0,
            }
        else:
            key = (payment_id, decision.chosen_intervention)
            if key not in potential_lookup.index:
                raise ValueError(f"Chosen potential outcome not found: {key}")
            selected = potential_lookup.loc[key]
            if int(selected["policy_allowed"]) != 1:
                raise ValueError(f"Logging policy selected blocked action: {key}")
            observed = {
                "recovered": int(selected["recovered"]),
                "amount_recovered": int(selected["amount_recovered"]),
                "intervention_cost": float(selected["intervention_cost"]),
                "net_recovered": float(selected["net_recovered"]),
                "time_to_recovery_hours": float(
                    selected["time_to_recovery_hours"]
                ),
            }
        record.update(observed)
        records.append(record)

    dataset = pd.DataFrame(records)
    dataset, cutoffs = _assign_temporal_splits(dataset, config)
    if FORBIDDEN_LOGGING_COLUMNS & set(dataset.columns):
        raise ValueError("Counterfactual simulator metadata leaked into logging data")
    return dataset, cutoffs


def validate_logging_dataset(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    dataset: pd.DataFrame,
    config: dict[str, Any],
    cutoffs: dict[str, str],
) -> dict[str, Any]:
    if len(dataset) != len(features):
        raise ValueError("Logging dataset must contain one row per failed payment")
    if dataset["payment_id"].duplicated().any():
        raise ValueError("Duplicate logged payments found")
    if set(dataset["payment_id"]) != set(features["transaction_id"]):
        raise ValueError("Logged payments differ from failed-payment features")
    valid_actions = set(config["eligible_interventions"]) | {
        str(config["fraud_action"])
    }
    if not set(dataset["chosen_intervention"]).issubset(valid_actions):
        raise ValueError("Invalid logged intervention found")
    if not dataset["policy_probability"].gt(0).all():
        raise ValueError("Non-positive propensity found")
    if not dataset["policy_probability"].le(1).all():
        raise ValueError("Propensity above one found")
    if FORBIDDEN_LOGGING_COLUMNS & set(dataset.columns):
        raise ValueError("Simulator answer exposed in logging dataset")

    fraud = dataset["fraud_flag"].eq(1)
    if not dataset.loc[fraud, "chosen_intervention"].eq(
        config["fraud_action"]
    ).all():
        raise ValueError("Fraud case received an automated action")
    if not dataset.loc[fraud, "recovered"].eq(0).all():
        raise ValueError("Fraud-blocked case recovered automatically")
    if dataset.loc[~fraud, "chosen_intervention"].eq(
        config["fraud_action"]
    ).any():
        raise ValueError("Eligible case incorrectly received no_action")

    exploration_rate = float(config["exploration_rate"])
    alternatives = len(config["eligible_interventions"]) - 1
    expected_propensities = {
        "behavior": 1 - exploration_rate,
        "exploration": exploration_rate / alternatives,
        "policy_blocked": 1.0,
    }
    for policy_type, expected in expected_propensities.items():
        rows = dataset["policy_type"].eq(policy_type)
        if rows.any() and not np.allclose(
            dataset.loc[rows, "policy_probability"],
            expected,
        ):
            raise ValueError(f"Incorrect propensity for {policy_type}")

    observed = dataset.loc[
        ~dataset["chosen_intervention"].eq(config["fraud_action"])
    ]
    selected_potential = outcomes.merge(
        observed[["payment_id", "chosen_intervention"]],
        left_on=["payment_id", "intervention"],
        right_on=["payment_id", "chosen_intervention"],
        how="inner",
        validate="one_to_one",
    ).set_index("payment_id")
    observed_lookup = observed.set_index("payment_id")
    for column in (
        "recovered",
        "amount_recovered",
        "intervention_cost",
        "net_recovered",
        "time_to_recovery_hours",
    ):
        if not np.allclose(
            observed_lookup.loc[selected_potential.index, column],
            selected_potential[column],
        ):
            raise ValueError(f"Observed {column} does not match potential outcome")

    timestamps = pd.to_datetime(dataset["prediction_time"], errors="raise")
    train = timestamps.loc[dataset["split"].eq("train")]
    validation = timestamps.loc[dataset["split"].eq("validation")]
    test = timestamps.loc[dataset["split"].eq("test")]
    if not train.max() < validation.min() or not validation.max() < test.min():
        raise ValueError("Temporal partitions overlap")

    eligible_rows = dataset.loc[~fraud]
    exploration_share = float(eligible_rows["policy_type"].eq("exploration").mean())
    if not 0.08 <= exploration_share <= 0.12:
        raise ValueError(f"Unexpected exploration share: {exploration_share}")

    intervention_counts = dataset["chosen_intervention"].value_counts()
    for intervention in config["eligible_interventions"]:
        if int(intervention_counts.get(intervention, 0)) < 100:
            raise ValueError(f"Insufficient support for {intervention}")

    eligible_logged = dataset.loc[~fraud, ["payment_id", "base_policy_intervention"]]
    base_outcomes = outcomes.merge(
        eligible_logged,
        left_on=["payment_id", "intervention"],
        right_on=["payment_id", "base_policy_intervention"],
        validate="one_to_one",
    )
    base_recoveries = int(base_outcomes["recovered"].sum())
    base_recovered_amount = int(base_outcomes["amount_recovered"].sum())

    retry_outcomes = outcomes.loc[
        outcomes["intervention"].eq("retry_payment")
        & outcomes["payment_id"].isin(eligible_logged["payment_id"])
    ]
    always_retry_recoveries = int(retry_outcomes["recovered"].sum())
    always_retry_amount = int(retry_outcomes["amount_recovered"].sum())

    oracle = (
        outcomes.loc[outcomes["policy_allowed"].eq(1)]
        .groupby("payment_id")
        .agg(
            recovered=("recovered", "max"),
            amount_recovered=("amount_recovered", "max"),
        )
    )

    return {
        "logged_payments": len(dataset),
        "unique_customers": int(dataset["customer_id"].nunique()),
        "action_counts": {
            key: int(value) for key, value in intervention_counts.items()
        },
        "policy_type_counts": {
            key: int(value)
            for key, value in dataset["policy_type"].value_counts().items()
        },
        "exploration_share_eligible": round(exploration_share, 6),
        "propensity_minimum": round(float(dataset["policy_probability"].min()), 6),
        "propensity_maximum": round(float(dataset["policy_probability"].max()), 6),
        "observed_recovery_rate": round(float(dataset["recovered"].mean()), 6),
        "observed_recovered_amount": int(dataset["amount_recovered"].sum()),
        "deterministic_base_policy_recovery_rate": round(
            base_recoveries / len(dataset),
            6,
        ),
        "deterministic_base_policy_recovered_amount": base_recovered_amount,
        "always_retry_recovery_rate": round(
            always_retry_recoveries / len(dataset),
            6,
        ),
        "always_retry_recovered_amount": always_retry_amount,
        "hindsight_oracle_upper_bound_recovery_rate": round(
            float(oracle["recovered"].sum()) / len(dataset),
            6,
        ),
        "hindsight_oracle_upper_bound_recovered_amount": int(
            oracle["amount_recovered"].sum()
        ),
        "split_counts": {
            key: int(value) for key, value in dataset["split"].value_counts().items()
        },
        "split_cutoffs": cutoffs,
        "simulator_probability_exposed": False,
        "counterfactual_outcomes_exposed": False,
        "outcome_integrity_violations": 0,
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    features_path: Path,
    outcomes_path: Path,
    config_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    features = pd.read_csv(features_path)
    outcomes = pd.read_csv(outcomes_path)
    config = load_policy_config(config_path)
    dataset, cutoffs = create_logging_dataset(features, outcomes, config)
    summary = validate_logging_dataset(
        features,
        outcomes,
        dataset,
        config,
        cutoffs,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.8f",
    )
    summary["features_sha256"] = _sha256(features_path)
    summary["potential_outcomes_sha256"] = _sha256(outcomes_path)
    summary["policy_config_sha256"] = _sha256(config_path)
    summary["logging_dataset_sha256"] = _sha256(output_path)
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== LOGGING POLICY SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved logging dataset: {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create one observed historical-policy outcome per payment."
    )
    parser.add_argument("--features", type=Path, default=DEFAULT_FEATURES_PATH)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.features,
        arguments.outcomes,
        arguments.config,
        arguments.output,
        arguments.summary_output,
    )
