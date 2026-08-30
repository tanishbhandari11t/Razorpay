from __future__ import annotations

import argparse
import json
import math
import platform
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn
import xgboost
from xgboost import XGBClassifier

from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    PlattCalibrator,
    build_model,
    build_preprocessor,
    calibration_bins,
    feature_columns,
    file_sha256,
    load_manifest,
    performance_by_intervention,
    predict_probabilities,
    probability_metrics,
    save_preprocessor,
    split_dataset,
    validate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
DEFAULT_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
DEFAULT_EVALUATION_DIR = REPO_ROOT / "ml" / "data" / "evaluation"
MODEL_VERSION = "v1"


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path: Path, value: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(value), indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )


def train(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    evaluation_dir: Path = DEFAULT_EVALUATION_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataframe = pd.read_csv(dataset_path)
    validate_dataset(dataframe, manifest, source_path=dataset_path)
    splits = split_dataset(dataframe, manifest)
    train_features, train_target = splits["train"]
    validation_features, validation_target = splits["validation"]
    test_features, test_target = splits["test"]

    preprocessor = build_preprocessor(manifest)
    train_matrix = preprocessor.fit_transform(train_features)
    validation_matrix = preprocessor.transform(validation_features)
    test_matrix = preprocessor.transform(test_features)

    model = build_model(seed=int(manifest["random_seed"]))
    model.fit(train_matrix, train_target)

    validation_raw = model.predict_proba(validation_matrix)[:, 1]
    calibrator = PlattCalibrator().fit(validation_raw, validation_target)
    validation_calibrated = calibrator.predict(validation_raw)
    test_raw = model.predict_proba(test_matrix)[:, 1]
    test_calibrated = calibrator.predict(test_raw)

    validation_frame = dataframe.loc[
        dataframe["split"].eq("validation")
    ].reset_index(drop=True)
    test_frame = dataframe.loc[dataframe["split"].eq("test")].reset_index(
        drop=True
    )
    metrics = {
        "model_version": MODEL_VERSION,
        "validation_before_calibration": probability_metrics(
            validation_target,
            validation_raw,
        ),
        "validation_after_platt_fit": probability_metrics(
            validation_target,
            validation_calibrated,
        ),
        "test_before_calibration": probability_metrics(
            test_target,
            test_raw,
        ),
        "test_after_calibration": probability_metrics(
            test_target,
            test_calibrated,
        ),
        "test_by_intervention": performance_by_intervention(
            test_frame,
            test_calibrated,
        ),
        "calibration": {
            "validation_before": calibration_bins(
                validation_target,
                validation_raw,
            ),
            "validation_after_platt_fit": calibration_bins(
                validation_target,
                validation_calibrated,
            ),
            "test_after": calibration_bins(test_target, test_calibrated),
        },
        "calibration_method": "Platt scaling fitted on validation only",
        "validation_after_platt_fit_is_resubstitution": True,
    }

    transformed_features = preprocessor.get_feature_names_out().tolist()
    importances = sorted(
        (
            {
                "feature": str(name),
                "importance": round(float(importance), 10),
            }
            for name, importance in zip(
                transformed_features,
                model.feature_importances_,
                strict=True,
            )
        ),
        key=lambda value: value["importance"],
        reverse=True,
    )

    artifact_dir.mkdir(parents=True, exist_ok=True)
    model_path = artifact_dir / "recovery_model_v1.json"
    preprocessor_path = artifact_dir / "preprocessing_v1.joblib"
    calibrator_path = artifact_dir / "calibration_v1.joblib"
    metrics_path = artifact_dir / "metrics_v1.json"
    metadata_path = artifact_dir / "model_metadata.json"
    importance_path = artifact_dir / "feature_importance_v1.json"

    model.save_model(model_path)
    save_preprocessor(preprocessor, preprocessor_path)
    joblib.dump(calibrator, calibrator_path)
    _write_json(metrics_path, metrics)
    _write_json(importance_path, importances)

    predictions = pd.concat(
        [
            validation_frame[
                ["payment_id", "customer_id", "prediction_time", "split",
                 "chosen_intervention", "recovered", "amount_inr"]
            ].assign(
                raw_recovery_probability=validation_raw,
                calibrated_recovery_probability=validation_calibrated,
            ),
            test_frame[
                ["payment_id", "customer_id", "prediction_time", "split",
                 "chosen_intervention", "recovered", "amount_inr"]
            ].assign(
                raw_recovery_probability=test_raw,
                calibrated_recovery_probability=test_calibrated,
            ),
        ],
        ignore_index=True,
    )
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = evaluation_dir / "observed_predictions_v1.csv"
    predictions.to_csv(predictions_path, index=False, float_format="%.8f")

    metadata = {
        "model_version": MODEL_VERSION,
        "dataset_name": manifest["dataset"]["name"],
        "dataset_version": manifest["dataset"]["version"],
        "dataset_sha256": file_sha256(dataset_path),
        "dataset_manifest_sha256": file_sha256(manifest_path),
        "features": feature_columns(manifest),
        "categorical_features": manifest["categorical_features"],
        "numerical_features": manifest["numerical_features"],
        "raw_feature_count": len(feature_columns(manifest)),
        "transformed_feature_count": len(transformed_features),
        "train_rows": len(train_features),
        "validation_rows": len(validation_features),
        "test_rows": len(test_features),
        "preprocessor_fit_split": "train",
        "model_fit_split": "train",
        "calibrator_fit_split": "validation",
        "test_used_for_fitting": False,
        "random_seed": int(manifest["random_seed"]),
        "model_parameters": model.get_params(),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "xgboost_version": xgboost.__version__,
        "artifacts": {
            "model": model_path.name,
            "preprocessor": preprocessor_path.name,
            "calibrator": calibrator_path.name,
            "metrics": metrics_path.name,
            "feature_importance": importance_path.name,
            "observed_predictions": str(predictions_path.relative_to(REPO_ROOT)),
        },
        "metrics": {
            "validation_after_platt_fit": metrics[
                "validation_after_platt_fit"
            ],
            "test_after_calibration": metrics["test_after_calibration"],
        },
    }
    _write_json(metadata_path, metadata)

    # Reload-path smoke check uses the same public inference function.
    reloaded = XGBClassifier()
    reloaded.load_model(model_path)
    reloaded_probabilities = predict_probabilities(
        preprocessor,
        reloaded,
        test_features,
        calibrator,
    )
    if not pd.Series(reloaded_probabilities).equals(
        pd.Series(test_calibrated)
    ):
        raise ValueError("Reloaded model predictions changed")

    print("=== RECOVERY MODEL V1 ===")
    print(json.dumps(metrics, indent=2, sort_keys=True))
    print(f"Saved artifacts: {artifact_dir}")
    return metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train and calibrate the RecoverAI XGBoost V1 model."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
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
    train(
        arguments.dataset,
        arguments.manifest,
        arguments.artifacts,
        arguments.evaluation_output,
    )
