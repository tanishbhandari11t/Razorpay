from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import yaml
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier

from app.services.features.builder import FEATURE_SCHEMA_PATH, model_feature_names


REPO_ROOT = Path(__file__).resolve().parents[3]
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
POLICY_V2_MANIFEST = REPO_ROOT / "ml" / "config" / "policy_v2_manifest.yaml"
POLICY_V3_MANIFEST = REPO_ROOT / "ml" / "config" / "policy_v3_manifest.yaml"
POLICY_EVALUATION_CONFIG = (
    REPO_ROOT / "ml" / "config" / "policy_evaluation.yaml"
)
MODEL_METADATA = ARTIFACT_DIR / "model_metadata.json"


class ArtifactValidationError(RuntimeError):
    pass


@dataclass(frozen=True)
class ModelBundle:
    model: XGBClassifier
    preprocessor: ColumnTransformer
    calibrator: Any
    model_version: str
    policy_version: str
    dataset_version: str
    raw_feature_names: tuple[str, ...]
    transformed_feature_count: int


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    v2 = yaml.safe_load(POLICY_V2_MANIFEST.read_text(encoding="utf-8"))
    v3 = yaml.safe_load(POLICY_V3_MANIFEST.read_text(encoding="utf-8"))
    evaluation = yaml.safe_load(
        POLICY_EVALUATION_CONFIG.read_text(encoding="utf-8")
    )
    metadata = json.loads(MODEL_METADATA.read_text(encoding="utf-8"))
    model_path = ARTIFACT_DIR / "recovery_model_v1.json"
    preprocessing_path = ARTIFACT_DIR / "preprocessing_v1.joblib"
    calibration_path = ARTIFACT_DIR / "calibration_v1.joblib"

    expected_model_hash = str(v2["model"]["sha256"])
    if _sha256(model_path) != expected_model_hash:
        raise ArtifactValidationError("Frozen recovery model hash changed")
    frozen = evaluation["frozen_artifacts"]
    frozen_files = {
        model_path: frozen["model_sha256"],
        preprocessing_path: frozen["preprocessor_sha256"],
        calibration_path: frozen["calibrator_sha256"],
        MODEL_METADATA: frozen["model_metadata_sha256"],
    }
    for path, expected_hash in frozen_files.items():
        if _sha256(path) != str(expected_hash):
            raise ArtifactValidationError(
                f"Frozen artifact hash changed: {path.name}"
            )
    expected_config_hash = v2["frozen_phase7"][
        "policy_evaluation_config_sha256"
    ]
    if _sha256(POLICY_EVALUATION_CONFIG) != str(expected_config_hash):
        raise ArtifactValidationError(
            "Frozen policy evaluation configuration changed"
        )
    if metadata["model_version"] != "v1":
        raise ArtifactValidationError("Unexpected recovery model version")
    if v3["model"]["version"] != "recovery_model_v1":
        raise ArtifactValidationError("Policy V3 references another model")
    if v3["policy"]["version"] != "recovery_policy_v3":
        raise ArtifactValidationError("Unexpected recovery policy version")
    if str(evaluation["dataset_version"]) != str(v3["dataset"]["version"]):
        raise ArtifactValidationError("Dataset versions do not match")
    contract = v3["online_feature_contract"]
    if _sha256(FEATURE_SCHEMA_PATH) != contract["feature_schema_sha256"]:
        raise ArtifactValidationError("Online feature contract hash changed")
    phase8_files = {
        REPO_ROOT / "ml" / "reports" / "phase8"
        / "phase8_report_manifest.json": v3["frozen_phase8"][
            "phase8_report_manifest_sha256"
        ],
        REPO_ROOT / "ml" / "config"
        / "intervention_policy.yaml": v3["frozen_phase8"][
            "intervention_policy_sha256"
        ],
        REPO_ROOT / "ml" / "config"
        / "action_matrix.yaml": v3["frozen_phase8"][
            "action_matrix_sha256"
        ],
        POLICY_V2_MANIFEST: v3["frozen_phase8"][
            "policy_v2_manifest_sha256"
        ],
    }
    for path, expected_hash in phase8_files.items():
        if _sha256(path) != str(expected_hash):
            raise ArtifactValidationError(
                f"Frozen Policy V3 input changed: {path.name}"
            )

    expected_features = tuple(model_feature_names())
    metadata_features = tuple(metadata["features"])
    if metadata_features != expected_features:
        raise ArtifactValidationError(
            "Online feature order differs from model metadata"
        )
    if int(metadata["raw_feature_count"]) != len(expected_features):
        raise ArtifactValidationError("Raw feature count changed")
    if len(expected_features) != int(contract["raw_model_feature_count"]):
        raise ArtifactValidationError("Feature contract raw count changed")

    preprocessor = joblib.load(preprocessing_path)
    transformed_count = len(preprocessor.get_feature_names_out())
    if transformed_count != int(metadata["transformed_feature_count"]):
        raise ArtifactValidationError("Preprocessor feature count changed")
    if transformed_count != int(contract["transformed_feature_count"]):
        raise ArtifactValidationError("Feature contract encoded count changed")
    model = XGBClassifier()
    model.load_model(model_path)
    if model.n_features_in_ != transformed_count:
        raise ArtifactValidationError(
            "Model and preprocessor encoded feature counts differ"
        )
    calibrator = joblib.load(calibration_path)
    return ModelBundle(
        model=model,
        preprocessor=preprocessor,
        calibrator=calibrator,
        model_version="recovery_model_v1",
        policy_version="recovery_policy_v3",
        dataset_version=str(v3["dataset"]["version"]),
        raw_feature_names=expected_features,
        transformed_feature_count=transformed_count,
    )
