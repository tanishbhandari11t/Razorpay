from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    feature_columns,
    load_manifest,
    load_preprocessor,
    predict_probabilities,
    validate_dataset,
)
from ml.src.policies.behavior_policy import load_policy_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
DEFAULT_OUTCOMES_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "intervention_outcomes.csv"
)
DEFAULT_LOGGING_SUMMARY_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_summary.json"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
DEFAULT_EVALUATION_DIR = REPO_ROOT / "ml" / "data" / "evaluation"


def _policy_outcomes(
    test: pd.DataFrame,
    potential: pd.DataFrame,
    actions: pd.Series,
) -> dict[str, float | int]:
    decisions = pd.DataFrame(
        {
            "payment_id": test["payment_id"].to_numpy(),
            "action": actions.to_numpy(),
            "amount_inr": test["amount_inr"].to_numpy(),
        }
    )
    automated = decisions.loc[decisions["action"].ne("no_action")]
    selected = automated.merge(
        potential,
        left_on=["payment_id", "action"],
        right_on=["payment_id", "intervention"],
        how="left",
        validate="one_to_one",
    )
    if selected["recovered"].isna().any():
        raise ValueError("A policy decision has no matching potential outcome")
    recoveries = int(selected["recovered"].sum())
    recovered_amount = int(selected["amount_recovered"].sum())
    return {
        "recovery_rate": round(recoveries / len(test), 8),
        "recovered_payments": recoveries,
        "recovered_amount": recovered_amount,
    }


def evaluate(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    evaluation_dir: Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)
    potential = pd.read_csv(outcomes_path)
    policy_config = load_policy_config()
    interventions = list(policy_config["eligible_interventions"])

    preprocessor = load_preprocessor(artifact_dir / "preprocessing_v1.joblib")
    calibrator = joblib.load(artifact_dir / "calibration_v1.joblib")
    model = XGBClassifier()
    model.load_model(artifact_dir / "recovery_model_v1.json")

    test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values(["prediction_time", "payment_id"])
        .reset_index(drop=True)
    )
    features = test[feature_columns(manifest)].copy()
    candidate_probabilities: dict[str, np.ndarray] = {}
    for intervention in interventions:
        candidate_features = features.copy()
        candidate_features["chosen_intervention"] = intervention
        candidate_probabilities[intervention] = predict_probabilities(
            preprocessor,
            model,
            candidate_features,
            calibrator,
        )

    score_matrix = np.column_stack(
        [candidate_probabilities[action] for action in interventions]
    )
    best_indices = np.argmax(score_matrix, axis=1)
    chosen = pd.Series(
        [interventions[index] for index in best_indices],
        index=test.index,
        dtype="string",
    )
    fraud = test["fraud_flag"].eq(1)
    chosen.loc[fraud] = "no_action"

    decisions = test[
        [
            "payment_id",
            "customer_id",
            "prediction_time",
            "amount_inr",
            "fraud_flag",
            "chosen_intervention",
            "recovered",
        ]
    ].rename(
        columns={
            "chosen_intervention": "logged_intervention",
            "recovered": "logged_recovered",
        }
    )
    decisions["recoverai_intervention"] = chosen
    decisions["recoverai_predicted_probability"] = np.max(score_matrix, axis=1)
    decisions.loc[fraud, "recoverai_predicted_probability"] = 0.0
    for intervention in interventions:
        decisions[f"predicted_{intervention}_probability"] = (
            candidate_probabilities[intervention]
        )

    always_retry = pd.Series("retry_payment", index=test.index, dtype="string")
    always_retry.loc[fraud] = "no_action"
    historical = test["base_policy_intervention"].astype("string")
    logged = test["chosen_intervention"].astype("string")

    oracle_potential = potential.loc[
        potential["payment_id"].isin(test["payment_id"])
        & potential["policy_allowed"].eq(1)
    ]
    oracle_index = (
        oracle_potential.sort_values(
            ["payment_id", "recovered", "amount_recovered", "intervention"],
            ascending=[True, False, False, True],
        )
        .drop_duplicates("payment_id")
        .set_index("payment_id")["intervention"]
    )
    oracle = test["payment_id"].map(oracle_index).fillna("no_action")

    policies = {
        "always_retry": _policy_outcomes(
            test,
            potential,
            always_retry,
        ),
        "historical_base_policy": _policy_outcomes(
            test,
            potential,
            historical,
        ),
        "logging_policy_observed": _policy_outcomes(
            test,
            potential,
            logged,
        ),
        "recoverai_v1": _policy_outcomes(
            test,
            potential,
            chosen,
        ),
        "hindsight_oracle_upper_bound": _policy_outcomes(
            test,
            potential,
            oracle,
        ),
    }
    recoverai = policies["recoverai_v1"]
    always_retry_result = policies["always_retry"]
    logging_summary = json.loads(
        DEFAULT_LOGGING_SUMMARY_PATH.read_text(encoding="utf-8")
    )
    result = {
        "model_version": "v1",
        "evaluation_rows": len(test),
        "evaluation_split": "frozen_test",
        "evaluation_scope": (
            "Synthetic counterfactual policy simulation; not real-world "
            "causal evidence"
        ),
        "selection_objective": (
            "Highest calibrated recovery probability; all intervention costs "
            "are currently zero"
        ),
        "policies": policies,
        "recoverai_action_counts": {
            key: int(value) for key, value in chosen.value_counts().items()
        },
        "recoverai_mean_predicted_recovery_probability": round(
            float(decisions["recoverai_predicted_probability"].mean()),
            8,
        ),
        "recoverai_uplift_vs_test_always_retry_percentage_points": round(
            100
            * (
                float(recoverai["recovery_rate"])
                - float(always_retry_result["recovery_rate"])
            ),
            4,
        ),
        "recoverai_recovered_amount_uplift_vs_test_always_retry": (
            int(recoverai["recovered_amount"])
            - int(always_retry_result["recovered_amount"])
        ),
        "recoverai_beats_test_always_retry": (
            float(recoverai["recovery_rate"])
            > float(always_retry_result["recovery_rate"])
        ),
        "full_dataset_reference_baselines": {
            "always_retry_recovery_rate": logging_summary[
                "always_retry_recovery_rate"
            ],
            "historical_base_policy_recovery_rate": logging_summary[
                "deterministic_base_policy_recovery_rate"
            ],
            "logging_policy_recovery_rate": logging_summary[
                "observed_recovery_rate"
            ],
        },
        "ips_or_doubly_robust_estimate_included": False,
    }

    evaluation_dir.mkdir(parents=True, exist_ok=True)
    decisions.to_csv(
        evaluation_dir / "test_policy_decisions_v1.csv",
        index=False,
        float_format="%.8f",
    )
    output_path = artifact_dir / "policy_evaluation_v1.json"
    output_path.write_text(
        json.dumps(result, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("=== RECOVERAI V1 POLICY EVALUATION ===")
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate RecoverAI V1 in the synthetic test environment."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--evaluation-output",
        type=Path,
        default=DEFAULT_EVALUATION_DIR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    evaluate(
        arguments.dataset,
        arguments.outcomes,
        arguments.manifest,
        arguments.artifacts,
        arguments.evaluation_output,
    )
