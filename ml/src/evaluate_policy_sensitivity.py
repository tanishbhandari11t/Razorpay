from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.analyze_support import DEFAULT_DATASET_PATH, DEFAULT_REPORT_DIR
from ml.src.evaluate_policy import _policy_outcomes
from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    file_sha256,
    load_manifest,
    membership_sha256,
    validate_dataset,
)
from ml.src.policies.support_safe_policy import load_support_policy_config
from ml.src.simulate_recovery import load_config as load_simulator_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTCOMES_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "intervention_outcomes.csv"
)
DEFAULT_V1_DECISIONS_PATH = (
    REPO_ROOT / "ml" / "data" / "evaluation" / "test_policy_decisions_v1.csv"
)


def _unit_interval(seed: int, *parts: object) -> float:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / 2**64


def perturb_intervention_probabilities(
    outcomes: pd.DataFrame,
    *,
    intervention: str,
    multiplier: float,
    seed: int,
) -> pd.DataFrame:
    perturbed = outcomes.copy()
    target = perturbed["intervention"].eq(intervention)
    perturbed.loc[target, "simulated_recovery_probability"] = np.clip(
        perturbed.loc[target, "simulated_recovery_probability"]
        * float(multiplier),
        0,
        1,
    )
    draws = np.array(
        [
            _unit_interval(seed, "outcome", payment_id, action)
            for payment_id, action in zip(
                perturbed["payment_id"],
                perturbed["intervention"],
                strict=True,
            )
        ]
    )
    perturbed["recovered"] = (
        perturbed["policy_allowed"].eq(1).to_numpy()
        & (
            draws
            < perturbed["simulated_recovery_probability"].to_numpy()
        )
    ).astype(int)
    perturbed["amount_recovered"] = (
        perturbed["amount_inr"] * perturbed["recovered"]
    )
    perturbed["net_recovered"] = (
        perturbed["amount_recovered"] - perturbed["intervention_cost"]
    )
    perturbed.loc[
        perturbed["recovered"].eq(0),
        "time_to_recovery_hours",
    ] = 0.0
    return perturbed


def evaluate_sensitivity(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    report_dir: Path = DEFAULT_REPORT_DIR,
    v1_decisions_path: Path = DEFAULT_V1_DECISIONS_PATH,
) -> dict[str, Any]:
    config = load_support_policy_config()
    simulator_config = load_simulator_config()
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)
    test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values(["prediction_time", "payment_id"])
        .reset_index(drop=True)
    )
    outcomes = pd.read_csv(outcomes_path)
    decisions = pd.read_csv(report_dir / "policy_decisions.csv")[
        ["payment_id", "selected_action"]
    ]
    v1_decisions = pd.read_csv(v1_decisions_path)[
        ["payment_id", "recoverai_intervention"]
    ]
    actions = test[["payment_id", "fraud_flag"]].merge(
        decisions,
        on="payment_id",
        validate="one_to_one",
    ).merge(
        v1_decisions,
        on="payment_id",
        validate="one_to_one",
    )
    fraud = actions["fraud_flag"].eq(1)
    always_retry = pd.Series(
        np.where(fraud, "no_action", "retry_payment"),
        dtype="string",
    )
    v1 = actions["recoverai_intervention"].astype("string")
    v2 = actions["selected_action"].astype("string")

    original = {
        "always_retry": _policy_outcomes(test, outcomes, always_retry),
        "recoverai_v1": _policy_outcomes(test, outcomes, v1),
        "recoverai_v2_support_safe": _policy_outcomes(
            test,
            outcomes,
            v2,
        ),
    }
    original["recoverai_v2_uplift_vs_always_retry_percentage_points"] = round(
        100
        * (
            float(original["recoverai_v2_support_safe"]["recovery_rate"])
            - float(original["always_retry"]["recovery_rate"])
        ),
        4,
    )

    sensitivity: list[dict[str, Any]] = []
    for intervention in config["interventions"]:
        for multiplier in config["sensitivity"][
            "intervention_probability_multipliers"
        ]:
            perturbed = perturb_intervention_probabilities(
                outcomes,
                intervention=intervention,
                multiplier=float(multiplier),
                seed=int(simulator_config["seed"]),
            )
            retry_result = _policy_outcomes(test, perturbed, always_retry)
            v2_result = _policy_outcomes(test, perturbed, v2)
            sensitivity.append(
                {
                    "perturbed_intervention": intervention,
                    "probability_multiplier": float(multiplier),
                    "always_retry_recovery_rate": retry_result[
                        "recovery_rate"
                    ],
                    "recoverai_v2_recovery_rate": v2_result[
                        "recovery_rate"
                    ],
                    "recoverai_v2_uplift_percentage_points": round(
                        100
                        * (
                            float(v2_result["recovery_rate"])
                            - float(retry_result["recovery_rate"])
                        ),
                        4,
                    ),
                    "recoverai_v2_beats_always_retry": (
                        float(v2_result["recovery_rate"])
                        > float(retry_result["recovery_rate"])
                    ),
                }
            )

    report = {
        "version": int(config["version"]),
        "scope": (
            "Synthetic counterfactual sensitivity analysis; simulator "
            "probabilities are environment-only and never model inputs"
        ),
        "frozen_test_rows": len(test),
        "original_environment": original,
        "intervention_sensitivity": sensitivity,
        "minimum_v2_uplift_percentage_points": min(
            value["recoverai_v2_uplift_percentage_points"]
            for value in sensitivity
        ),
        "maximum_v2_uplift_percentage_points": max(
            value["recoverai_v2_uplift_percentage_points"]
            for value in sensitivity
        ),
        "scenarios_where_v2_beats_always_retry": sum(
            value["recoverai_v2_beats_always_retry"]
            for value in sensitivity
        ),
        "total_sensitivity_scenarios": len(sensitivity),
    }
    synthetic_path = report_dir / "synthetic_policy_evaluation.json"
    synthetic_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    report_files = {
        "support_report": report_dir / "support_report.json",
        "context_support": report_dir / "context_support.csv",
        "offline_policy_metrics": report_dir / "offline_policy_metrics.json",
        "policy_decisions": report_dir / "policy_decisions.csv",
        "synthetic_policy_evaluation": synthetic_path,
    }
    phase7_manifest = {
        "version": 1,
        "model_version": config["model_version"],
        "dataset_version": config["dataset_version"],
        "frozen_test_rows": len(test),
        "frozen_test_membership_sha256": membership_sha256(
            test["payment_id"]
        ),
        "policy_evaluation_config_sha256": file_sha256(
            REPO_ROOT / "ml" / "config" / "policy_evaluation.yaml"
        ),
        "report_sha256": {
            name: file_sha256(path) for name, path in report_files.items()
        },
        "original_synthetic_v2_recovery_rate": original[
            "recoverai_v2_support_safe"
        ]["recovery_rate"],
        "original_synthetic_always_retry_recovery_rate": original[
            "always_retry"
        ]["recovery_rate"],
        "sensitivity_scenarios_won": report[
            "scenarios_where_v2_beats_always_retry"
        ],
        "sensitivity_scenarios_total": report[
            "total_sensitivity_scenarios"
        ],
    }
    (report_dir / "phase7_report_manifest.json").write_text(
        json.dumps(phase7_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("=== SYNTHETIC POLICY SENSITIVITY ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate support-safe policy under simulator perturbations."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
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
    evaluate_sensitivity(
        arguments.dataset,
        arguments.outcomes,
        arguments.manifest,
        arguments.reports,
        arguments.v1_decisions,
    )
