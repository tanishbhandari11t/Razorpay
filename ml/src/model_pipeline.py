from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from scipy.special import expit, logit
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST_PATH = REPO_ROOT / "ml" / "config" / "dataset_manifest.yaml"

POST_TREATMENT_COLUMNS = {
    "recovered",
    "amount_recovered",
    "intervention_cost",
    "net_recovered",
    "time_to_recovery_hours",
}


@dataclass
class PlattCalibrator:
    model: LogisticRegression | None = None

    def fit(
        self,
        probabilities: np.ndarray,
        target: pd.Series | np.ndarray,
    ) -> "PlattCalibrator":
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        self.model = LogisticRegression(random_state=42)
        self.model.fit(logit(clipped).reshape(-1, 1), np.asarray(target))
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise ValueError("Calibrator has not been fitted")
        clipped = np.clip(np.asarray(probabilities), 1e-6, 1 - 1e-6)
        calibrated_logit = self.model.decision_function(
            logit(clipped).reshape(-1, 1)
        )
        return expit(calibrated_logit)


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def membership_sha256(payment_ids: pd.Series) -> str:
    payload = "\n".join(sorted(payment_ids.astype(str))) + "\n"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def feature_columns(manifest: dict[str, Any]) -> list[str]:
    return [
        *manifest["numerical_features"],
        *manifest["categorical_features"],
    ]


def validate_dataset(
    dataframe: pd.DataFrame,
    manifest: dict[str, Any],
    source_path: Path | None = None,
) -> None:
    if source_path is not None:
        actual_hash = file_sha256(source_path)
        expected_hash = str(manifest["dataset"]["sha256"])
        if actual_hash != expected_hash:
            raise ValueError(
                f"Frozen dataset hash changed: expected {expected_hash}, "
                f"got {actual_hash}"
            )
    if len(dataframe) != int(manifest["dataset"]["rows"]):
        raise ValueError("Frozen dataset row count changed")
    if dataframe["payment_id"].duplicated().any():
        raise ValueError("Duplicate payment IDs found")
    if set(dataframe["split"]) != {"train", "validation", "test"}:
        raise ValueError("Unexpected frozen split labels")

    target = str(manifest["target"]["name"])
    if set(dataframe[target].unique()) - {0, 1}:
        raise ValueError("Recovery target must be binary")
    expected_features = feature_columns(manifest)
    missing = set(expected_features) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing model features: {sorted(missing)}")
    if target in expected_features:
        raise ValueError("Target leaked into model features")
    if POST_TREATMENT_COLUMNS & set(expected_features):
        raise ValueError("Post-treatment outcome leaked into model features")

    lowered_features = [column.lower() for column in expected_features]
    for forbidden in manifest["forbidden_patterns"]:
        if any(str(forbidden).lower() in column for column in lowered_features):
            raise ValueError(f"Forbidden model feature pattern: {forbidden}")

    for split_name in ("train", "validation", "test"):
        rows = dataframe.loc[dataframe["split"].eq(split_name)]
        specification = manifest["split"][split_name]
        if len(rows) != int(specification["rows"]):
            raise ValueError(f"Frozen {split_name} row count changed")
        if membership_sha256(rows["payment_id"]) != specification["membership_sha256"]:
            raise ValueError(f"Frozen {split_name} membership changed")

    timestamps = pd.to_datetime(dataframe["prediction_time"], errors="raise")
    train = timestamps.loc[dataframe["split"].eq("train")]
    validation = timestamps.loc[dataframe["split"].eq("validation")]
    test = timestamps.loc[dataframe["split"].eq("test")]
    if not train.max() < validation.min() or not validation.max() < test.min():
        raise ValueError("Temporal split ordering changed")


def split_dataset(
    dataframe: pd.DataFrame,
    manifest: dict[str, Any],
) -> dict[str, tuple[pd.DataFrame, pd.Series]]:
    columns = feature_columns(manifest)
    target = str(manifest["target"]["name"])
    return {
        split_name: (
            dataframe.loc[dataframe["split"].eq(split_name), columns].copy(),
            dataframe.loc[dataframe["split"].eq(split_name), target].copy(),
        )
        for split_name in ("train", "validation", "test")
    }


def build_preprocessor(manifest: dict[str, Any]) -> ColumnTransformer:
    return ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", manifest["numerical_features"]),
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=True,
                    dtype=np.float32,
                ),
                manifest["categorical_features"],
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_model(seed: int = 42) -> XGBClassifier:
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=seed,
        n_jobs=-1,
        tree_method="hist",
    )


def predict_probabilities(
    preprocessor: ColumnTransformer,
    model: XGBClassifier,
    features: pd.DataFrame,
    calibrator: PlattCalibrator | None = None,
) -> np.ndarray:
    raw = model.predict_proba(preprocessor.transform(features))[:, 1]
    return calibrator.predict(raw) if calibrator is not None else raw


def probability_metrics(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
) -> dict[str, float | int | None]:
    actual = np.asarray(target)
    predictions = np.asarray(probabilities)
    has_both_classes = len(np.unique(actual)) == 2
    observed_bins, predicted_bins = calibration_curve(
        actual,
        predictions,
        n_bins=min(10, len(actual)),
        strategy="quantile",
    )
    result: dict[str, float | int | None] = {
        "rows": len(actual),
        "positive_rate": round(float(actual.mean()), 8),
        "pr_auc": (
            round(float(average_precision_score(actual, predictions)), 8)
            if has_both_classes
            else None
        ),
        "log_loss": round(
            float(log_loss(actual, predictions, labels=[0, 1])),
            8,
        ),
        "brier_score": round(float(brier_score_loss(actual, predictions)), 8),
        "mean_absolute_calibration_error": round(
            float(np.mean(np.abs(observed_bins - predicted_bins))),
            8,
        ),
    }
    result["roc_auc"] = (
        round(float(roc_auc_score(actual, predictions)), 8)
        if has_both_classes
        else None
    )
    return result


def calibration_bins(
    target: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    bins: int = 10,
) -> list[dict[str, float]]:
    observed, predicted = calibration_curve(
        target,
        probabilities,
        n_bins=bins,
        strategy="quantile",
    )
    return [
        {
            "mean_predicted_probability": round(float(prediction), 8),
            "observed_recovery_rate": round(float(actual), 8),
        }
        for prediction, actual in zip(predicted, observed, strict=True)
    ]


def performance_by_intervention(
    frame: pd.DataFrame,
    probabilities: np.ndarray,
) -> dict[str, dict[str, float | int | None]]:
    scored = frame.copy()
    scored["_probability"] = probabilities
    return {
        str(intervention): probability_metrics(
            rows["recovered"],
            rows["_probability"].to_numpy(),
        )
        for intervention, rows in scored.groupby("chosen_intervention")
    }


def save_preprocessor(
    preprocessor: ColumnTransformer,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(preprocessor, path)


def load_preprocessor(path: Path) -> ColumnTransformer:
    return joblib.load(path)
