from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import yaml
from xgboost import XGBClassifier

from ml.src.model_pipeline import (
    feature_columns,
    file_sha256,
    load_manifest,
    validate_dataset,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "online_model_schema.yaml"
AVAILABILITY_PATH = (
    REPO_ROOT / "ml" / "config" / "online_feature_schema.yaml"
)
DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "online_training_features.csv"
)
MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
MODEL_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_model_v2_online_manifest.yaml"
)
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase13"


def test_online_model_schema_has_33_sourced_features() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    availability = yaml.safe_load(
        AVAILABILITY_PATH.read_text(encoding="utf-8")
    )
    features = schema["features"]
    names = [feature["name"] for feature in features]
    deployable = {
        *availability["deployable_model_features"]["online_available"],
        *availability["deployable_model_features"]["online_derivable"],
    }
    assert len(names) == len(set(names)) == 33
    assert set(names) == deployable
    assert all(feature["source"] for feature in features)
    assert all(feature["temporal_cutoff"] for feature in features)
    assert schema["real_shadow_training_allowed"] is False


def test_unsupported_features_are_not_model_inputs() -> None:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    manifest = load_manifest(MANIFEST_PATH)
    model_features = set(feature_columns(manifest))
    removed = set(schema["unsupported_features"]["remove"])
    assert len(removed) == 16
    assert model_features.isdisjoint(removed)
    assert "chosen_intervention" in model_features


def test_v2_dataset_preserves_temporal_membership() -> None:
    dataframe = pd.read_csv(DATASET_PATH)
    manifest = load_manifest(MANIFEST_PATH)
    validate_dataset(dataframe, manifest, source_path=DATASET_PATH)
    assert len(dataframe) == 12376
    assert manifest["dataset"]["provenance"] == "SYNTHETIC"
    assert manifest["dataset"]["real_shadow_cases_included"] is False
    assert dataframe["split"].value_counts().to_dict() == {
        "train": 8663,
        "validation": 1856,
        "test": 1857,
    }


def test_v2_artifacts_are_isolated_and_reloadable() -> None:
    model_manifest = yaml.safe_load(
        MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    assert model_manifest["model"]["raw_feature_count"] == 33
    assert model_manifest["training"]["real_shadow_cases_used"] is False
    for entry in [
        *model_manifest["artifacts"].values(),
        *model_manifest["frozen_inputs"].values(),
    ]:
        assert file_sha256(REPO_ROOT / entry["path"]) == entry["sha256"]
    artifact_dir = REPO_ROOT / "ml" / "artifacts" / "v2_online"
    preprocessor = joblib.load(
        artifact_dir / "preprocessing_v2_online.joblib"
    )
    model = XGBClassifier()
    model.load_model(artifact_dir / "recovery_model_v2_online.json")
    assert len(preprocessor.get_feature_names_out()) == model.n_features_in_
    assert model.n_features_in_ == 43


def test_phase13_reports_preserve_evidence_boundaries() -> None:
    readiness = json.loads(
        (REPORT_DIR / "phase13_readiness.json").read_text(encoding="utf-8")
    )
    replay = json.loads(
        (REPORT_DIR / "v2_online_shadow_replay.json").read_text(
            encoding="utf-8"
        )
    )
    assert readiness["model_ready"] is False
    assert readiness["execution"]["controlled"] == "BLOCKED"
    assert readiness["execution"]["provider_actions_enabled"] is False
    assert replay["training_allowed"] is False
    assert replay["recovery_uplift_claim_allowed"] is False
    assert replay["cases"] == 20
    assert replay["compatibility_allowed_cases"] == 0
    assert replay["safety"]["provider_calls"] == 0
    assert replay["safety"]["database_writes"] == 0


def test_phase13_manifest_hashes_all_outputs() -> None:
    manifest = json.loads(
        (REPORT_DIR / "phase13_report_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    for path, entry in {
        **manifest["inputs"],
        **manifest["artifacts"],
    }.items():
        assert file_sha256(REPO_ROOT / path) == entry["sha256"]
    assert manifest["policy_v3_modified"] is False
    assert manifest["real_shadow_training_allowed"] is False
    assert manifest["promotion"]["controlled_execution_authorized"] is False
