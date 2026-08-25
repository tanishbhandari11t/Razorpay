from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "failed_payment_features.csv"
)
DEFAULT_CONFIG_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_simulation.yaml"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "intervention_outcomes.csv"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "recovery_simulation_summary.json"
)

EXPECTED_INTERVENTIONS = [
    "retry_payment",
    "payment_link",
    "whatsapp_reminder",
    "escalate_to_merchant",
]
REQUIRED_FEATURES = {
    "transaction_id",
    "customer_id",
    "amount_inr",
    "fraud_flag",
    "has_prior_history",
    "has_previous_success",
    "historical_success_rate",
    "previous_transaction_count",
    "previous_failure_streak",
    "failures_last_7d",
    "failures_last_30d",
    "transactions_last_30d",
    "days_since_previous_success",
    "amount_vs_previous_avg",
    "device_matches_primary",
    "network_matches_primary",
}


def load_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    interventions = list(config.get("interventions", {}))
    if interventions != EXPECTED_INTERVENTIONS:
        raise ValueError(
            f"Expected interventions {EXPECTED_INTERVENTIONS}, got {interventions}"
        )
    scenario_probabilities = config.get("scenario_probabilities", {})
    if not math.isclose(sum(scenario_probabilities.values()), 1.0, abs_tol=1e-9):
        raise ValueError("Scenario probabilities must sum to 1")
    probability = config.get("probability", {})
    if not 0 <= probability["minimum"] < probability["maximum"] <= 1:
        raise ValueError("Invalid probability bounds")
    return config


def _validate_features(features: pd.DataFrame) -> None:
    missing = REQUIRED_FEATURES - set(features.columns)
    if missing:
        raise ValueError(f"Missing simulator features: {sorted(missing)}")
    if features["transaction_id"].duplicated().any():
        raise ValueError("Duplicate failed-payment feature rows found")
    if features.isna().any().any():
        raise ValueError("Missing values found in failed-payment features")


def _unit_interval(seed: int, *parts: object) -> float:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / 2**64


def _normal(seed: int, *parts: object) -> float:
    value = np.clip(_unit_interval(seed, *parts), 1e-12, 1 - 1e-12)
    return NormalDist().inv_cdf(float(value))


def _sigmoid(score: float) -> float:
    if score >= 0:
        return 1 / (1 + math.exp(-score))
    exponential = math.exp(score)
    return exponential / (1 + exponential)


def _synthetic_scenario(
    transaction_id: str,
    config: dict[str, Any],
) -> str:
    draw = _unit_interval(config["seed"], "scenario", transaction_id)
    cumulative = 0.0
    for scenario, probability in config["scenario_probabilities"].items():
        cumulative += float(probability)
        if draw < cumulative:
            return str(scenario)
    return str(next(reversed(config["scenario_probabilities"])))


def _observable_score(row: pd.Series, config: dict[str, Any]) -> dict[str, float]:
    probability = config["probability"]
    coefficients = probability["observable_coefficients"]
    success = coefficients["historical_success_rate"]
    amount_ratio = float(np.clip(row["amount_vs_previous_avg"], 0, 10))
    log_amount_ratio = math.log1p(amount_ratio)
    log_previous_count = math.log1p(max(float(row["previous_transaction_count"]), 0))
    log_transactions_30d = math.log1p(
        max(float(row["transactions_last_30d"]), 0)
    )

    if int(row["has_previous_success"]):
        days_since_success = max(float(row["days_since_previous_success"]), 0)
        recent_success = math.exp(
            -days_since_success / coefficients["recent_success_decay_days"]
        )
        no_success_adjustment = 0.0
    else:
        recent_success = 0.0
        no_success_adjustment = coefficients["no_previous_success_adjustment"]

    score = float(probability["base_score"])
    score += success["coefficient"] * (
        float(row["historical_success_rate"]) - success["center"]
    )
    score += coefficients["log_previous_transaction_count"] * log_previous_count
    score += coefficients["previous_failure_streak"] * float(
        row["previous_failure_streak"]
    )
    score += coefficients["failures_last_7d"] * float(row["failures_last_7d"])
    score += coefficients["failures_last_30d"] * float(row["failures_last_30d"])
    score += coefficients["log_transactions_last_30d"] * log_transactions_30d
    score += coefficients["log_amount_vs_previous_avg"] * log_amount_ratio
    score += coefficients["device_matches_primary"] * float(
        row["device_matches_primary"]
    )
    score += coefficients["network_matches_primary"] * float(
        row["network_matches_primary"]
    )
    score += coefficients["has_prior_history"] * float(row["has_prior_history"])
    score += coefficients["recent_success_strength"] * recent_success
    score += no_success_adjustment

    return {
        "score": score,
        "log_amount_ratio": log_amount_ratio,
        "log_previous_count": log_previous_count,
        "log_transactions_30d": log_transactions_30d,
        "recent_success": recent_success,
    }


def _intervention_score(
    row: pd.Series,
    intervention: str,
    scenario: str,
    transformed: dict[str, float],
    config: dict[str, Any],
) -> float:
    intervention_config = config["interventions"][intervention]
    score = float(intervention_config["base_score"])
    score += float(intervention_config["scenario_adjustments"][scenario])
    interactions = intervention_config.get("interactions", {})

    if "recent_success_strength" in interactions:
        if int(row["has_previous_success"]):
            days = max(float(row["days_since_previous_success"]), 0)
            recent = math.exp(
                -days / float(interactions["recent_success_decay_days"])
            )
        else:
            recent = 0.0
        score += float(interactions["recent_success_strength"]) * recent
    score += float(interactions.get("previous_failure_streak", 0)) * float(
        row["previous_failure_streak"]
    )
    score += float(interactions.get("historical_success_rate", 0)) * float(
        row["historical_success_rate"]
    )
    score += float(interactions.get("log_amount_vs_previous_avg", 0)) * transformed[
        "log_amount_ratio"
    ]
    score += float(interactions.get("log_transactions_last_30d", 0)) * transformed[
        "log_transactions_30d"
    ]
    score += float(interactions.get("has_prior_history", 0)) * float(
        row["has_prior_history"]
    )
    return score


def _is_policy_allowed(
    row: pd.Series,
    intervention: str,
    config: dict[str, Any],
) -> bool:
    blocked = set(config["policy"]["fraud_flag_blocks"])
    return not (int(row["fraud_flag"]) == 1 and intervention in blocked)


def _recovery_time(
    transaction_id: str,
    intervention: str,
    config: dict[str, Any],
) -> float:
    assumptions = config["interventions"][intervention]["recovery_time_hours"]
    draw = _unit_interval(
        config["seed"],
        "recovery-time",
        transaction_id,
        intervention,
    )
    shaped = draw ** float(assumptions["shape"])
    value = float(assumptions["minimum"]) + shaped * (
        float(assumptions["maximum"]) - float(assumptions["minimum"])
    )
    return round(value, 4)


def simulate_outcomes(
    features: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    _validate_features(features)
    seed = int(config["seed"])
    hidden = config["probability"]["hidden_variation"]
    minimum_probability = float(config["probability"]["minimum"])
    maximum_probability = float(config["probability"]["maximum"])
    rows: list[dict[str, Any]] = []

    ordered_features = features.sort_values(
        ["customer_id", "prediction_time", "transaction_id"],
        kind="mergesort",
    )
    for _, feature in ordered_features.iterrows():
        transaction_id = str(feature["transaction_id"])
        customer_id = str(feature["customer_id"])
        scenario = _synthetic_scenario(transaction_id, config)
        transformed = _observable_score(feature, config)
        customer_responsiveness = _normal(
            seed,
            "customer-responsiveness",
            customer_id,
        ) * float(hidden["customer_responsiveness_std"])
        payment_shock = _normal(
            seed,
            "payment-shock",
            transaction_id,
        ) * float(hidden["payment_specific_shock_std"])

        for intervention in EXPECTED_INTERVENTIONS:
            allowed = _is_policy_allowed(feature, intervention, config)
            preference = _normal(
                seed,
                "customer-intervention-preference",
                customer_id,
                intervention,
            ) * float(hidden["customer_intervention_preference_std"])
            score = transformed["score"]
            score += customer_responsiveness + payment_shock + preference
            score += _intervention_score(
                feature,
                intervention,
                scenario,
                transformed,
                config,
            )
            unconstrained_probability = float(
                np.clip(
                    _sigmoid(score),
                    minimum_probability,
                    maximum_probability,
                )
            )
            probability = unconstrained_probability if allowed else 0.0
            outcome_draw = _unit_interval(
                seed,
                "outcome",
                transaction_id,
                intervention,
            )
            recovered = int(allowed and outcome_draw < probability)
            amount = int(feature["amount_inr"])
            cost = float(config["intervention_costs"][intervention])
            amount_recovered = amount if recovered else 0

            rows.append(
                {
                    "payment_id": transaction_id,
                    "customer_id": customer_id,
                    "intervention": intervention,
                    "synthetic_failure_scenario": scenario,
                    "policy_allowed": int(allowed),
                    "simulated_recovery_probability": probability,
                    "recovered": recovered,
                    "amount_inr": amount,
                    "amount_recovered": amount_recovered,
                    "intervention_cost": cost,
                    "net_recovered": amount_recovered - cost,
                    "time_to_recovery_hours": (
                        _recovery_time(transaction_id, intervention, config)
                        if recovered
                        else 0.0
                    ),
                    "simulation_version": int(config["version"]),
                }
            )

    outcomes = pd.DataFrame(rows)
    if outcomes.isna().any().any():
        raise ValueError("Simulation produced missing values")
    return outcomes


def validate_simulation(
    features: pd.DataFrame,
    outcomes: pd.DataFrame,
    config: dict[str, Any],
) -> dict[str, Any]:
    expected_rows = len(features) * len(EXPECTED_INTERVENTIONS)
    if len(outcomes) != expected_rows:
        raise ValueError(f"Expected {expected_rows} outcomes, got {len(outcomes)}")
    if outcomes.duplicated(["payment_id", "intervention"]).any():
        raise ValueError("Duplicate payment/intervention outcomes found")
    actions_per_payment = outcomes.groupby("payment_id")["intervention"].nunique()
    if not actions_per_payment.eq(len(EXPECTED_INTERVENTIONS)).all():
        raise ValueError("A payment is missing potential interventions")
    if not outcomes["recovered"].isin({0, 1}).all():
        raise ValueError("Invalid simulated outcomes found")
    if not outcomes.loc[outcomes["recovered"].eq(0), "amount_recovered"].eq(0).all():
        raise ValueError("Unrecovered cases contain recovered money")
    recovered = outcomes["recovered"].eq(1)
    if not outcomes.loc[recovered, "amount_recovered"].eq(
        outcomes.loc[recovered, "amount_inr"]
    ).all():
        raise ValueError("Recovered amount does not match payment amount")
    if not outcomes.loc[~outcomes["policy_allowed"].astype(bool), "recovered"].eq(
        0
    ).all():
        raise ValueError("Policy-blocked intervention was executed")

    allowed = outcomes.loc[outcomes["policy_allowed"].eq(1)].copy()
    allowed["probability_bucket"] = pd.cut(
        allowed["simulated_recovery_probability"],
        bins=np.linspace(0, 1, 11),
        include_lowest=True,
    )
    calibration = (
        allowed.groupby("probability_bucket", observed=True)
        .agg(
            rows=("recovered", "size"),
            mean_probability=("simulated_recovery_probability", "mean"),
            actual_recovery_rate=("recovered", "mean"),
        )
        .reset_index()
    )
    calibration["absolute_gap"] = (
        calibration["mean_probability"] - calibration["actual_recovery_rate"]
    ).abs()
    calibration_error = float(
        np.average(calibration["absolute_gap"], weights=calibration["rows"])
    )
    if calibration_error > 0.03:
        raise ValueError(f"Simulator calibration error is too high: {calibration_error}")

    best = (
        allowed.sort_values(
            ["payment_id", "simulated_recovery_probability", "intervention"],
            ascending=[True, False, True],
            kind="mergesort",
        )
        .drop_duplicates("payment_id")
        .groupby("intervention")
        .size()
        .reindex(EXPECTED_INTERVENTIONS, fill_value=0)
    )
    if best.max() / best.sum() >= 0.70:
        raise ValueError("One intervention dominates at least 70% of best actions")
    if best.eq(0).any():
        raise ValueError("At least one intervention is never the best action")

    joined = allowed.merge(
        features[
            [
                "transaction_id",
                "historical_success_rate",
                "has_prior_history",
                "amount_vs_previous_avg",
            ]
        ],
        left_on="payment_id",
        right_on="transaction_id",
        validate="many_to_one",
    )
    high_quality = joined.loc[
        joined["has_prior_history"].eq(1)
        & joined["historical_success_rate"].ge(0.90),
        "recovered",
    ].mean()
    low_quality = joined.loc[
        joined["has_prior_history"].eq(0)
        | joined["historical_success_rate"].lt(0.70),
        "recovered",
    ].mean()
    if not high_quality > low_quality:
        raise ValueError("Customer history does not affect simulated recovery")

    intervention_rates = (
        outcomes.groupby("intervention")
        .agg(
            rows=("recovered", "size"),
            policy_allowed_rate=("policy_allowed", "mean"),
            mean_probability=("simulated_recovery_probability", "mean"),
            recovery_rate=("recovered", "mean"),
            recovered_amount=("amount_recovered", "sum"),
        )
        .reindex(EXPECTED_INTERVENTIONS)
    )
    scenario_counts = outcomes.drop_duplicates("payment_id")[
        "synthetic_failure_scenario"
    ].value_counts()

    return {
        "failed_payments": len(features),
        "potential_outcomes": len(outcomes),
        "interventions_per_payment": len(EXPECTED_INTERVENTIONS),
        "policy_blocked_rows": int(outcomes["policy_allowed"].eq(0).sum()),
        "overall_recovery_rate": round(float(outcomes["recovered"].mean()), 6),
        "allowed_recovery_rate": round(float(allowed["recovered"].mean()), 6),
        "calibration_error": round(calibration_error, 6),
        "high_quality_recovery_rate": round(float(high_quality), 6),
        "low_quality_recovery_rate": round(float(low_quality), 6),
        "best_intervention_counts": {
            key: int(value) for key, value in best.items()
        },
        "intervention_metrics": {
            intervention: {
                key: round(float(value), 6)
                for key, value in metrics.items()
            }
            for intervention, metrics in intervention_rates.to_dict(
                orient="index"
            ).items()
        },
        "scenario_counts": {
            key: int(value) for key, value in scenario_counts.items()
        },
        "calibration_buckets": [
            {
                "bucket": str(row["probability_bucket"]),
                "rows": int(row["rows"]),
                "mean_probability": round(float(row["mean_probability"]), 6),
                "actual_recovery_rate": round(
                    float(row["actual_recovery_rate"]),
                    6,
                ),
            }
            for _, row in calibration.iterrows()
        ],
        "maximum_raw_amount_ratio": round(
            float(features["amount_vs_previous_avg"].max()),
            6,
        ),
        "maximum_simulator_log_amount_ratio": round(math.log1p(10), 6),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    input_path: Path,
    config_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    features = pd.read_csv(input_path)
    config = load_config(config_path)
    outcomes = simulate_outcomes(features, config)
    summary = validate_simulation(features, outcomes, config)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    outcomes.to_csv(
        output_path,
        index=False,
        float_format="%.8f",
    )
    summary["input_sha256"] = _sha256(input_path)
    summary["config_sha256"] = _sha256(config_path)
    summary["outcomes_sha256"] = _sha256(output_path)
    summary["simulator_probabilities_are_model_features"] = False
    summary["costs_and_recovery_times_are_synthetic_assumptions"] = True
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== RECOVERY SIMULATION SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved outcomes: {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate deterministic potential recovery outcomes."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(
        arguments.input,
        arguments.config,
        arguments.output,
        arguments.summary_output,
    )
