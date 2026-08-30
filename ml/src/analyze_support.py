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
    file_sha256,
    load_manifest,
    load_preprocessor,
    predict_probabilities,
    validate_dataset,
)
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    build_context_support_table,
    load_support_policy_config,
    select_support_safe_action,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports"

FROZEN_PATHS = {
    "dataset_sha256": DEFAULT_DATASET_PATH,
    "model_sha256": DEFAULT_ARTIFACT_DIR / "recovery_model_v1.json",
    "preprocessor_sha256": DEFAULT_ARTIFACT_DIR / "preprocessing_v1.joblib",
    "calibrator_sha256": DEFAULT_ARTIFACT_DIR / "calibration_v1.joblib",
    "model_metadata_sha256": DEFAULT_ARTIFACT_DIR / "model_metadata.json",
    "metrics_sha256": DEFAULT_ARTIFACT_DIR / "metrics_v1.json",
    "policy_evaluation_v1_sha256": (
        DEFAULT_ARTIFACT_DIR / "policy_evaluation_v1.json"
    ),
    "policy_decisions_v1_sha256": (
        REPO_ROOT / "ml" / "data" / "evaluation" / "test_policy_decisions_v1.csv"
    ),
}


def verify_frozen_v1(
    config: dict[str, Any],
    paths: dict[str, Path] = FROZEN_PATHS,
) -> dict[str, str]:
    verified: dict[str, str] = {}
    for key, path in paths.items():
        actual = file_sha256(path)
        expected = str(config["frozen_artifacts"][key])
        if actual != expected:
            raise ValueError(
                f"Frozen Phase 6 artifact changed: {path.name}; "
                f"expected {expected}, got {actual}"
            )
        verified[key] = actual
    return verified


def _distribution(values: pd.Series) -> dict[str, float]:
    return {
        "minimum": round(float(values.min()), 8),
        "p05": round(float(values.quantile(0.05)), 8),
        "p25": round(float(values.quantile(0.25)), 8),
        "median": round(float(values.median()), 8),
        "p75": round(float(values.quantile(0.75)), 8),
        "p95": round(float(values.quantile(0.95)), 8),
        "maximum": round(float(values.max()), 8),
    }


def _load_frozen_model(artifact_dir: Path):
    preprocessor = load_preprocessor(artifact_dir / "preprocessing_v1.joblib")
    calibrator = joblib.load(artifact_dir / "calibration_v1.joblib")
    model = XGBClassifier()
    model.load_model(artifact_dir / "recovery_model_v1.json")
    return preprocessor, model, calibrator


def build_candidate_probabilities(
    dataframe: pd.DataFrame,
    manifest: dict[str, Any],
    config: dict[str, Any],
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
) -> dict[str, np.ndarray]:
    preprocessor, model, calibrator = _load_frozen_model(artifact_dir)
    base_features = dataframe[feature_columns(manifest)].copy()
    probabilities: dict[str, np.ndarray] = {}
    for action in config["interventions"]:
        candidates = base_features.copy()
        candidates["chosen_intervention"] = action
        probabilities[action] = predict_probabilities(
            preprocessor,
            model,
            candidates,
            calibrator,
        )
    return probabilities


def analyze(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> tuple[dict[str, Any], pd.DataFrame]:
    config = load_support_policy_config()
    frozen_hashes = verify_frozen_v1(config)
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)

    preprocessor, model, calibrator = _load_frozen_model(artifact_dir)
    observed_features = dataset[feature_columns(manifest)]
    dataset["_observed_action_probability"] = predict_probabilities(
        preprocessor,
        model,
        observed_features,
        calibrator,
    )

    action_report: dict[str, Any] = {}
    for action in [*config["interventions"], "no_action"]:
        action_rows = dataset.loc[dataset["chosen_intervention"].eq(action)]
        if action_rows.empty:
            continue
        split_metrics = {}
        for split_name in ("train", "validation", "test"):
            rows = action_rows.loc[action_rows["split"].eq(split_name)]
            split_metrics[split_name] = {
                "count": len(rows),
                "observed_recovery_rate": (
                    round(float(rows["recovered"].mean()), 8)
                    if len(rows)
                    else None
                ),
                "mean_predicted_probability": (
                    round(
                        float(rows["_observed_action_probability"].mean()),
                        8,
                    )
                    if len(rows)
                    else None
                ),
            }
        action_report[action] = {
            "splits": split_metrics,
            "total_count": len(action_rows),
            "observed_recovery_rate": round(
                float(action_rows["recovered"].mean()),
                8,
            ),
            "mean_predicted_probability": round(
                float(action_rows["_observed_action_probability"].mean()),
                8,
            ),
            "predicted_probability_distribution": _distribution(
                action_rows["_observed_action_probability"]
            ),
        }

    training = dataset.loc[dataset["split"].eq("train")]
    support_table = build_context_support_table(training, config)
    support_index = SupportIndex(support_table, config)
    test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values(["prediction_time", "payment_id"])
        .reset_index(drop=True)
    )
    probabilities = build_candidate_probabilities(
        test,
        manifest,
        config,
        artifact_dir,
    )

    decision_rows: list[dict[str, Any]] = []
    for index, row in test.iterrows():
        candidate_probabilities = {
            action: float(probabilities[action][index])
            for action in config["interventions"]
        }
        decision = select_support_safe_action(
            row,
            candidate_probabilities,
            support_index,
            config,
        )
        output = {
            "payment_id": row["payment_id"],
            "customer_id": row["customer_id"],
            "prediction_time": row["prediction_time"],
            "logged_action": row["chosen_intervention"],
            "logging_probability": row["policy_probability"],
            "logged_recovered": row["recovered"],
            "base_policy_intervention": row["base_policy_intervention"],
            "fraud_flag": row["fraud_flag"],
            "raw_best_action": decision.raw_best_action,
            "selected_action": decision.selected_action,
            "selected_probability": decision.selected_probability,
            "decision_margin": decision.decision_margin,
            "fallback_used": decision.fallback_used,
            "decision_reason": decision.decision_reason,
            "selected_action_supported": decision.selected_action_supported,
            "candidate_actions": decision.candidate_summary_json(
                candidate_probabilities
            ),
        }
        for action in config["interventions"]:
            evidence = decision.candidate_evidence[action]
            output[f"predicted_{action}_probability"] = candidate_probabilities[
                action
            ]
            output[f"{action}_support_count"] = evidence.action_count
            output[f"{action}_support_ess"] = evidence.effective_sample_size
            output[f"{action}_supported"] = evidence.supported
        decision_rows.append(output)
    decisions = pd.DataFrame(decision_rows)

    probability_columns = [
        f"predicted_{action}_probability"
        for action in config["interventions"]
    ]
    probability_frame = decisions[probability_columns]
    ranks = probability_frame.rank(axis=1, ascending=False, method="min")
    retry_column = "predicted_retry_payment_probability"
    escalation_column = "predicted_escalate_to_merchant_probability"
    sorted_probabilities = np.sort(probability_frame.to_numpy(), axis=1)
    raw_margins = sorted_probabilities[:, -1] - sorted_probabilities[:, -2]

    report = {
        "version": int(config["version"]),
        "model_version": config["model_version"],
        "frozen_artifacts_verified": frozen_hashes,
        "support_thresholds": config["support"],
        "threshold_derivation": {
            "positive_context_action_cells": len(support_table),
            "median_positive_cell_count": round(
                float(support_table["action_count"].median()),
                8,
            ),
            "median_context_effective_sample_size": round(
                float(support_table["effective_sample_size"].median()),
                8,
            ),
            "supported_context_action_cells": int(
                support_table["supported"].sum()
            ),
        },
        "actions": action_report,
        "context_support": support_table.to_dict(orient="records"),
        "test_candidate_support_rates": {
            action: round(
                float(decisions[f"{action}_supported"].mean()),
                8,
            )
            for action in config["interventions"]
        },
        "support_safe_policy": {
            "selected_action_counts": {
                str(key): int(value)
                for key, value in decisions["selected_action"]
                .value_counts()
                .items()
            },
            "fallback_count": int(decisions["fallback_used"].sum()),
            "fallback_rate": round(
                float(decisions["fallback_used"].mean()),
                8,
            ),
            "decision_reason_counts": {
                str(key): int(value)
                for key, value in decisions["decision_reason"]
                .value_counts()
                .items()
            },
            "unsupported_model_selection_count": int(
                (
                    ~decisions["selected_action_supported"]
                    & ~decisions["decision_reason"].isin(
                        [
                            "fraud_policy_block",
                            "no_supported_candidate_safety_fallback",
                        ]
                    )
                ).sum()
            ),
        },
        "retry_investigation": {
            "raw_selection_count": int(
                decisions["raw_best_action"].eq("retry_payment").sum()
            ),
            "rank_counts": {
                str(int(key)): int(value)
                for key, value in ranks[retry_column].value_counts().items()
            },
            "mean_predicted_probability": round(
                float(decisions[retry_column].mean()),
                8,
            ),
            "within_validation_calibration_margin_of_best_count": int(
                (
                    probability_frame.max(axis=1)
                    - decisions[retry_column]
                    <= float(
                        config["decision"]["minimum_probability_margin"]
                    )
                ).sum()
            ),
            "raw_best_margin_distribution": _distribution(
                pd.Series(raw_margins)
            ),
        },
        "escalation_investigation": {
            "raw_selection_count": int(
                decisions["raw_best_action"].eq(
                    "escalate_to_merchant"
                ).sum()
            ),
            "rank_counts": {
                str(int(key)): int(value)
                for key, value in ranks[escalation_column].value_counts().items()
            },
            "mean_predicted_probability": round(
                float(decisions[escalation_column].mean()),
                8,
            ),
            "mean_gap_below_best": round(
                float(
                    (
                        probability_frame.max(axis=1)
                        - decisions[escalation_column]
                    ).mean()
                ),
                8,
            ),
        },
        "counterfactual_outcomes_used": False,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    support_table.to_csv(report_dir / "context_support.csv", index=False)
    decisions.to_csv(
        report_dir / "policy_decisions.csv",
        index=False,
        float_format="%.8f",
    )
    (report_dir / "support_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("=== PHASE 7 SUPPORT REPORT ===")
    print(json.dumps(report["support_safe_policy"], indent=2, sort_keys=True))
    return report, decisions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze intervention support and create V2 decisions."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument("--reports", type=Path, default=DEFAULT_REPORT_DIR)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    analyze(
        arguments.dataset,
        arguments.manifest,
        arguments.artifacts,
        arguments.reports,
    )
