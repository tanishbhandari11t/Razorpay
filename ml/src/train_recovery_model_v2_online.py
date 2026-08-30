from __future__ import annotations

import json
import math
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
import yaml
from xgboost import XGBClassifier

from ml.src.model_pipeline import (
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
DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "online_training_features.csv"
)
DATASET_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
ONLINE_MODEL_SCHEMA_PATH = (
    REPO_ROOT / "ml" / "config" / "online_model_schema.yaml"
)
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "v2_online"
EVALUATION_DIR = REPO_ROOT / "ml" / "data" / "evaluation"
MODEL_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_model_v2_online_manifest.yaml"
)
PHASE11_BASELINE_PATH = (
    REPO_ROOT / "ml" / "config" / "phase11_baseline_manifest.yaml"
)
PHASE12_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase12" / "phase12_manifest.json"
)
MODEL_VERSION = "recovery_model_v2_online"
ARTIFACT_VERSION = "v2_online"


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
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def train_v2_online() -> dict[str, Any]:
    manifest = load_manifest(DATASET_MANIFEST_PATH)
    dataframe = pd.read_csv(DATASET_PATH)
    validate_dataset(dataframe, manifest, source_path=DATASET_PATH)
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
        "objective": "P(recovery | deployable_context, candidate_action)",
        "evidence_source": "synthetic_frozen_temporal_split",
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
            "test_after": calibration_bins(
                test_target,
                test_calibrated,
            ),
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

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    model_path = ARTIFACT_DIR / "recovery_model_v2_online.json"
    preprocessor_path = ARTIFACT_DIR / "preprocessing_v2_online.joblib"
    calibrator_path = ARTIFACT_DIR / "calibration_v2_online.joblib"
    metrics_path = ARTIFACT_DIR / "metrics_v2_online.json"
    metadata_path = ARTIFACT_DIR / "model_metadata_v2_online.json"
    importance_path = ARTIFACT_DIR / "feature_importance_v2_online.json"

    model.save_model(model_path)
    save_preprocessor(preprocessor, preprocessor_path)
    joblib.dump(calibrator, calibrator_path)
    _write_json(metrics_path, metrics)
    _write_json(importance_path, importances)

    prediction_columns = [
        "payment_id",
        "customer_id",
        "prediction_time",
        "split",
        "chosen_intervention",
        "recovered",
        "amount_inr",
    ]
    predictions = pd.concat(
        [
            validation_frame[prediction_columns].assign(
                raw_recovery_probability=validation_raw,
                calibrated_recovery_probability=validation_calibrated,
            ),
            test_frame[prediction_columns].assign(
                raw_recovery_probability=test_raw,
                calibrated_recovery_probability=test_calibrated,
            ),
        ],
        ignore_index=True,
    )
    EVALUATION_DIR.mkdir(parents=True, exist_ok=True)
    predictions_path = (
        EVALUATION_DIR / "observed_predictions_v2_online.csv"
    )
    predictions.to_csv(predictions_path, index=False, float_format="%.8f")

    metadata = {
        "model_version": MODEL_VERSION,
        "artifact_version": ARTIFACT_VERSION,
        "dataset_name": manifest["dataset"]["name"],
        "dataset_version": manifest["dataset"]["version"],
        "dataset_sha256": file_sha256(DATASET_PATH),
        "dataset_manifest_sha256": file_sha256(DATASET_MANIFEST_PATH),
        "online_model_schema_sha256": file_sha256(ONLINE_MODEL_SCHEMA_PATH),
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
        "real_shadow_cases_used_for_fitting": False,
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
            "observed_predictions": str(
                predictions_path.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
        },
        "metrics": {
            "validation_after_platt_fit": metrics[
                "validation_after_platt_fit"
            ],
            "test_after_calibration": metrics["test_after_calibration"],
        },
    }
    _write_json(metadata_path, metadata)

    reloaded = XGBClassifier()
    reloaded.load_model(model_path)
    reloaded_probabilities = predict_probabilities(
        preprocessor,
        reloaded,
        test_features,
        calibrator,
    )
    if not np.array_equal(reloaded_probabilities, test_calibrated):
        raise ValueError("Reloaded V2-online model predictions changed")

    artifacts = {
        path.name: {
            "path": str(path.relative_to(REPO_ROOT)).replace("\\", "/"),
            "sha256": file_sha256(path),
        }
        for path in (
            model_path,
            preprocessor_path,
            calibrator_path,
            metrics_path,
            metadata_path,
            importance_path,
            predictions_path,
        )
    }
    model_manifest = {
        "version": 1,
        "model": {
            "version": MODEL_VERSION,
            "artifact_version": ARTIFACT_VERSION,
            "objective": metrics["objective"],
            "raw_feature_count": len(feature_columns(manifest)),
            "transformed_feature_count": len(transformed_features),
        },
        "generated_at": datetime.now(UTC).isoformat(),
        "training": {
            "dataset_provenance": "SYNTHETIC",
            "real_shadow_cases_used": False,
            "temporal_split": True,
            "test_used_for_fitting": False,
        },
        "frozen_inputs": {
            "dataset_manifest_v2_online": {
                "path": str(
                    DATASET_MANIFEST_PATH.relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "sha256": file_sha256(DATASET_MANIFEST_PATH),
            },
            "online_model_schema": {
                "path": str(
                    ONLINE_MODEL_SCHEMA_PATH.relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "sha256": file_sha256(ONLINE_MODEL_SCHEMA_PATH),
            },
            "phase11_baseline": {
                "path": str(
                    PHASE11_BASELINE_PATH.relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "sha256": file_sha256(PHASE11_BASELINE_PATH),
            },
            "phase12_manifest": {
                "path": str(
                    PHASE12_MANIFEST_PATH.relative_to(REPO_ROOT)
                ).replace("\\", "/"),
                "sha256": file_sha256(PHASE12_MANIFEST_PATH),
            },
        },
        "artifacts": artifacts,
        "deployment": {
            "shadow_authorized": False,
            "controlled_execution_authorized": False,
            "provider_actions_enabled": False,
        },
    }
    MODEL_MANIFEST_PATH.write_text(
        yaml.safe_dump(model_manifest, sort_keys=False),
        encoding="utf-8",
    )
    return {
        "metrics": metrics,
        "metadata": metadata,
        "manifest": model_manifest,
    }


if __name__ == "__main__":
    result = train_v2_online()
    print(json.dumps(result["metrics"], indent=2, sort_keys=True))
