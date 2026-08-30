from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.analyze_support import (
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_DIR,
    verify_frozen_v1,
)
from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    validate_dataset,
)
from ml.src.policies.support_safe_policy import load_support_policy_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_V1_DECISIONS_PATH = (
    REPO_ROOT / "ml" / "data" / "evaluation" / "test_policy_decisions_v1.csv"
)


def clipped_propensities(
    propensities: np.ndarray,
    minimum_propensity: float,
) -> np.ndarray:
    values = np.asarray(propensities, dtype=float)
    if not 0 < minimum_propensity <= 1:
        raise ValueError("Minimum propensity must be in (0, 1]")
    if np.any(values <= 0) or np.any(values > 1):
        raise ValueError("Observed propensities must be in (0, 1]")
    return np.maximum(values, minimum_propensity)


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=float)
    denominator = np.square(values).sum()
    return float(values.sum() ** 2 / denominator) if denominator else 0.0


def ips_estimate(
    rewards: np.ndarray,
    target_probability_for_logged_action: np.ndarray,
    logging_propensity: np.ndarray,
    minimum_propensity: float,
) -> tuple[float, float, np.ndarray]:
    denominator = clipped_propensities(
        logging_propensity,
        minimum_propensity,
    )
    weights = (
        np.asarray(target_probability_for_logged_action, dtype=float)
        / denominator
    )
    ips = float(np.mean(weights * np.asarray(rewards, dtype=float)))
    weight_sum = weights.sum()
    snips = (
        float(np.sum(weights * rewards) / weight_sum)
        if weight_sum > 0
        else float("nan")
    )
    return ips, snips, weights


def doubly_robust_estimate(
    rewards: np.ndarray,
    predicted_logged_reward: np.ndarray,
    predicted_target_reward: np.ndarray,
    target_probability_for_logged_action: np.ndarray,
    logging_propensity: np.ndarray,
    minimum_propensity: float,
) -> tuple[float, np.ndarray]:
    denominator = clipped_propensities(
        logging_propensity,
        minimum_propensity,
    )
    weights = (
        np.asarray(target_probability_for_logged_action, dtype=float)
        / denominator
    )
    contributions = np.asarray(predicted_target_reward, dtype=float) + weights * (
        np.asarray(rewards, dtype=float)
        - np.asarray(predicted_logged_reward, dtype=float)
    )
    return float(contributions.mean()), contributions


@dataclass(frozen=True)
class PolicyArrays:
    target_probability_for_logged_action: np.ndarray
    predicted_target_reward: np.ndarray
    target_actions: pd.Series | None


def _predicted_for_actions(
    decisions: pd.DataFrame,
    actions: pd.Series,
    interventions: list[str],
) -> np.ndarray:
    result = np.zeros(len(decisions), dtype=float)
    for action in interventions:
        mask = actions.eq(action).to_numpy()
        result[mask] = decisions.loc[
            mask,
            f"predicted_{action}_probability",
        ]
    return result


def _deterministic_policy_arrays(
    decisions: pd.DataFrame,
    actions: pd.Series,
    interventions: list[str],
) -> PolicyArrays:
    return PolicyArrays(
        target_probability_for_logged_action=(
            actions.to_numpy() == decisions["logged_action"].to_numpy()
        ).astype(float),
        predicted_target_reward=_predicted_for_actions(
            decisions,
            actions,
            interventions,
        ),
        target_actions=actions,
    )


def _logging_policy_arrays(
    decisions: pd.DataFrame,
    interventions: list[str],
) -> PolicyArrays:
    base = decisions["base_policy_intervention"].astype("string")
    fraud = decisions["fraud_flag"].eq(1)
    predicted = np.zeros(len(decisions), dtype=float)
    exploration_probability = 0.1 / (len(interventions) - 1)
    for action in interventions:
        action_prediction = decisions[
            f"predicted_{action}_probability"
        ].to_numpy()
        probability = np.where(
            base.eq(action),
            0.9,
            exploration_probability,
        )
        predicted += probability * action_prediction
    predicted[fraud.to_numpy()] = 0.0
    return PolicyArrays(
        target_probability_for_logged_action=decisions[
            "logging_probability"
        ].to_numpy(float),
        predicted_target_reward=predicted,
        target_actions=None,
    )


def _cluster_bootstrap(
    customers: pd.Series,
    ips_contributions: np.ndarray,
    dr_contributions: np.ndarray,
    weights: np.ndarray,
    weighted_rewards: np.ndarray,
    *,
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, list[float]]:
    codes, unique_customers = pd.factorize(customers, sort=True)
    groups = len(unique_customers)
    row_counts = np.bincount(codes)
    ips_sums = np.bincount(codes, weights=ips_contributions)
    dr_sums = np.bincount(codes, weights=dr_contributions)
    weight_sums = np.bincount(codes, weights=weights)
    weighted_reward_sums = np.bincount(codes, weights=weighted_rewards)
    random = np.random.default_rng(seed)
    ips_values = np.empty(iterations)
    dr_values = np.empty(iterations)
    snips_values = np.empty(iterations)
    for iteration in range(iterations):
        sampled = random.integers(0, groups, size=groups)
        sampled_rows = row_counts[sampled].sum()
        ips_values[iteration] = ips_sums[sampled].sum() / sampled_rows
        dr_values[iteration] = dr_sums[sampled].sum() / sampled_rows
        sampled_weight = weight_sums[sampled].sum()
        snips_values[iteration] = (
            weighted_reward_sums[sampled].sum() / sampled_weight
            if sampled_weight > 0
            else np.nan
        )
    alpha = (1 - confidence_level) / 2

    def interval(values: np.ndarray) -> list[float]:
        return [
            round(float(np.nanquantile(values, alpha)), 8),
            round(float(np.nanquantile(values, 1 - alpha)), 8),
        ]

    return {
        "ips": interval(ips_values),
        "self_normalized_ips": interval(snips_values),
        "doubly_robust": interval(dr_values),
    }


def _evaluate_policy(
    name: str,
    arrays: PolicyArrays,
    decisions: pd.DataFrame,
    minimum_propensity: float,
    config: dict[str, Any],
) -> dict[str, Any]:
    rewards = decisions["logged_recovered"].to_numpy(float)
    logging_propensity = decisions["logging_probability"].to_numpy(float)
    logged_actions = decisions["logged_action"].astype("string")
    predicted_logged = _predicted_for_actions(
        decisions,
        logged_actions,
        list(config["interventions"]),
    )
    ips, snips, weights = ips_estimate(
        rewards,
        arrays.target_probability_for_logged_action,
        logging_propensity,
        minimum_propensity,
    )
    dr, dr_contributions = doubly_robust_estimate(
        rewards,
        predicted_logged,
        arrays.predicted_target_reward,
        arrays.target_probability_for_logged_action,
        logging_propensity,
        minimum_propensity,
    )
    weighted_rewards = weights * rewards
    intervals = _cluster_bootstrap(
        decisions["customer_id"],
        weighted_rewards,
        dr_contributions,
        weights,
        weighted_rewards,
        iterations=int(
            config["offline_evaluation"]["bootstrap_iterations"]
        ),
        confidence_level=float(
            config["offline_evaluation"]["confidence_level"]
        ),
        seed=int(config["random_seed"]),
    )
    positive_weights = weights > 0
    return {
        "policy": name,
        "ips": round(ips, 8),
        "self_normalized_ips": round(snips, 8),
        "doubly_robust": round(dr, 8),
        "confidence_intervals_95": intervals,
        "matching_logged_rows": int(positive_weights.sum()),
        "matching_rate": round(float(positive_weights.mean()), 8),
        "importance_weight_effective_sample_size": round(
            effective_sample_size(weights),
            8,
        ),
        "maximum_importance_weight": round(float(weights.max()), 8),
        "mean_predicted_target_reward": round(
            float(arrays.predicted_target_reward.mean()),
            8,
        ),
    }


def evaluate_offline(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    v1_decisions_path: Path = DEFAULT_V1_DECISIONS_PATH,
) -> dict[str, Any]:
    config = load_support_policy_config()
    verify_frozen_v1(config)
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)
    test = dataset.loc[dataset["split"].eq("test")].copy()

    decisions_path = report_dir / "policy_decisions.csv"
    if not decisions_path.exists():
        raise FileNotFoundError("Run `python -m ml.src.analyze_support` first")
    decisions = pd.read_csv(decisions_path)
    expected_columns = {
        "payment_id",
        "logged_action",
        "logging_probability",
        "logged_recovered",
        "selected_action",
    }
    if expected_columns - set(decisions.columns):
        raise ValueError("Support-safe decisions are missing required columns")
    if any("outcome" in column.lower() for column in decisions.columns):
        raise ValueError("Counterfactual outcome leaked into policy decisions")
    if set(decisions["payment_id"]) != set(test["payment_id"]):
        raise ValueError("Policy decisions changed the frozen test membership")

    v1_decisions = pd.read_csv(v1_decisions_path)[
        ["payment_id", "recoverai_intervention"]
    ]
    decisions = decisions.merge(
        v1_decisions,
        on="payment_id",
        how="left",
        validate="one_to_one",
    )
    interventions = list(config["interventions"])
    fraud = decisions["fraud_flag"].eq(1)
    always_retry = pd.Series(
        np.where(fraud, "no_action", "retry_payment"),
        dtype="string",
    )
    historical_base = decisions["base_policy_intervention"].astype("string")
    recoverai_v1 = decisions["recoverai_intervention"].astype("string")
    recoverai_v2 = decisions["selected_action"].astype("string")

    policy_arrays = {
        "always_retry": _deterministic_policy_arrays(
            decisions,
            always_retry,
            interventions,
        ),
        "historical_base_policy": _deterministic_policy_arrays(
            decisions,
            historical_base,
            interventions,
        ),
        "logging_policy": _logging_policy_arrays(decisions, interventions),
        "recoverai_v1": _deterministic_policy_arrays(
            decisions,
            recoverai_v1,
            interventions,
        ),
        "recoverai_v2_support_safe": _deterministic_policy_arrays(
            decisions,
            recoverai_v2,
            interventions,
        ),
    }
    minimum_propensity = 1 / float(
        config["offline_evaluation"]["maximum_importance_weight"]
    )
    results = {
        name: _evaluate_policy(
            name,
            arrays,
            decisions,
            minimum_propensity,
            config,
        )
        for name, arrays in policy_arrays.items()
    }

    sensitivity: dict[str, Any] = {}
    for threshold in config["offline_evaluation"]["clipping_sensitivity"]:
        sensitivity[str(threshold)] = {
            name: {
                key: value
                for key, value in _evaluate_policy(
                    name,
                    arrays,
                    decisions,
                    float(threshold),
                    {
                        **config,
                        "offline_evaluation": {
                            **config["offline_evaluation"],
                            "bootstrap_iterations": 1,
                        },
                    },
                ).items()
                if key
                in {
                    "ips",
                    "self_normalized_ips",
                    "doubly_robust",
                    "importance_weight_effective_sample_size",
                    "maximum_importance_weight",
                }
            }
            for name, arrays in policy_arrays.items()
        }

    report = {
        "version": int(config["version"]),
        "evaluation_split": "frozen_test",
        "rows": len(decisions),
        "minimum_propensity": minimum_propensity,
        "maximum_importance_weight": config["offline_evaluation"][
            "maximum_importance_weight"
        ],
        "bootstrap": {
            "iterations": config["offline_evaluation"][
                "bootstrap_iterations"
            ],
            "unit": config["offline_evaluation"]["bootstrap_unit"],
            "confidence_level": config["offline_evaluation"][
                "confidence_level"
            ],
            "seed": config["random_seed"],
        },
        "policies": results,
        "clipping_sensitivity": sensitivity,
        "logged_observed_recovery_rate": round(
            float(decisions["logged_recovered"].mean()),
            8,
        ),
        "counterfactual_outcomes_used": False,
        "simulator_probability_used": False,
        "interpretation": (
            "IPS and doubly robust estimates use only the observed logged "
            "action, observed recovery, logging propensity, and frozen V1 "
            "reward predictions."
        ),
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "offline_policy_metrics.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("=== OFFLINE POLICY EVALUATION ===")
    print(json.dumps(results, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run IPS and doubly robust policy evaluation."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument(
        "--v1-decisions",
        type=Path,
        default=DEFAULT_V1_DECISIONS_PATH,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate_offline(
        arguments.dataset,
        arguments.manifest,
        arguments.reports,
        arguments.v1_decisions,
    )
