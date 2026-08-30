from __future__ import annotations

import json
import sys
from functools import lru_cache
from pathlib import Path

import joblib
import yaml
from xgboost import XGBClassifier

from app.ml.model_loader import (
    ArtifactValidationError,
    ModelBundle,
    _sha256,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "v2_online"
MODEL_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_model_v2_online_manifest.yaml"
)
DATASET_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
ONLINE_MODEL_SCHEMA_PATH = (
    REPO_ROOT / "ml" / "config" / "online_model_schema.yaml"
)
METADATA_PATH = ARTIFACT_DIR / "model_metadata_v2_online.json"


@lru_cache(maxsize=1)
def load_v2_online_model_bundle() -> ModelBundle:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    model_manifest = yaml.safe_load(
        MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    dataset_manifest = yaml.safe_load(
        DATASET_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    online_schema = yaml.safe_load(
        ONLINE_MODEL_SCHEMA_PATH.read_text(encoding="utf-8")
    )
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    artifact_entries = model_manifest["artifacts"]
    for entry in artifact_entries.values():
        path = REPO_ROOT / entry["path"]
        if _sha256(path) != str(entry["sha256"]):
            raise ArtifactValidationError(
                f"V2-online artifact hash changed: {path.name}"
            )
    frozen_inputs = model_manifest["frozen_inputs"]
    for entry in frozen_inputs.values():
        path = REPO_ROOT / entry["path"]
        if _sha256(path) != str(entry["sha256"]):
            raise ArtifactValidationError(
                f"V2-online frozen input changed: {path.name}"
            )
    if metadata["model_version"] != "recovery_model_v2_online":
        raise ArtifactValidationError("Unexpected V2-online model version")
    expected_features = tuple(
        [
            *dataset_manifest["numerical_features"],
            *dataset_manifest["categorical_features"],
        ]
    )
    schema_features = tuple(
        feature["name"] for feature in online_schema["features"]
    )
    if set(expected_features) != set(schema_features):
        raise ArtifactValidationError(
            "V2-online model manifest and online schema differ"
        )
    if tuple(metadata["features"]) != expected_features:
        raise ArtifactValidationError(
            "V2-online feature order differs from metadata"
        )
    if len(expected_features) != 33:
        raise ArtifactValidationError("V2-online raw feature count changed")

    preprocessor = joblib.load(
        ARTIFACT_DIR / "preprocessing_v2_online.joblib"
    )
    transformed_count = len(preprocessor.get_feature_names_out())
    if transformed_count != int(metadata["transformed_feature_count"]):
        raise ArtifactValidationError(
            "V2-online encoded feature count changed"
        )
    model = XGBClassifier()
    model.load_model(ARTIFACT_DIR / "recovery_model_v2_online.json")
    if model.n_features_in_ != transformed_count:
        raise ArtifactValidationError(
            "V2-online model and preprocessor dimensions differ"
        )
    calibrator = joblib.load(
        ARTIFACT_DIR / "calibration_v2_online.joblib"
    )
    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        calibrator=calibrator,
        model_version="recovery_model_v2_online",
        policy_version="recovery_policy_v3_v2_online",
        dataset_version=str(dataset_manifest["dataset"]["version"]),
        raw_feature_names=expected_features,
        transformed_feature_count=transformed_count,
    )
