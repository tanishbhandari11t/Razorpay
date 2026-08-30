from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from ml.src.model_pipeline import file_sha256, membership_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
SOURCE_DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
SOURCE_MANIFEST_PATH = REPO_ROOT / "ml" / "config" / "dataset_manifest.yaml"
ONLINE_SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "online_model_schema.yaml"
OUTPUT_DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "online_training_features.csv"
)
OUTPUT_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
REPORT_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase13" / "feature_disposition.json"
)

IDENTIFIERS = ["payment_id", "customer_id", "prediction_time", "split"]
TARGET = "recovered"
EVALUATION_METADATA = [
    "fraud_flag",
    "base_policy_intervention",
    "policy_probability",
    "policy_type",
    "logging_policy_version",
    "amount_recovered",
    "intervention_cost",
    "net_recovered",
    "time_to_recovery_hours",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_online_training_dataset() -> dict[str, Any]:
    source_manifest = _load_yaml(SOURCE_MANIFEST_PATH)
    schema = _load_yaml(ONLINE_SCHEMA_PATH)
    source = pd.read_csv(SOURCE_DATASET_PATH)
    features = [feature["name"] for feature in schema["features"]]
    if len(features) != 33 or len(set(features)) != 33:
        raise ValueError("V2-online schema must contain 33 unique features")
    if set(schema["unsupported_features"]["remove"]) & set(features):
        raise ValueError("Unsupported features leaked into V2-online schema")
    required = {
        *IDENTIFIERS,
        *features,
        TARGET,
        *EVALUATION_METADATA,
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(f"Source dataset is missing columns: {sorted(missing)}")
    if source["payment_id"].duplicated().any():
        raise ValueError("Source dataset contains duplicate payment IDs")
    if set(source["split"]) != {"train", "validation", "test"}:
        raise ValueError("Source split membership is invalid")

    ordered = [
        *IDENTIFIERS,
        *features,
        TARGET,
        *EVALUATION_METADATA,
    ]
    output = source.loc[:, ordered].copy()
    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(OUTPUT_DATASET_PATH, index=False, float_format="%.8f")

    numerical = [
        feature["name"]
        for feature in schema["features"]
        if feature["type"] == "numeric"
    ]
    categorical = [
        feature["name"]
        for feature in schema["features"]
        if feature["type"] == "categorical"
    ]
    split_manifest = {}
    for split_name in ("train", "validation", "test"):
        rows = output.loc[output["split"].eq(split_name)]
        split_manifest[split_name] = {
            "rows": len(rows),
            "membership_sha256": membership_sha256(rows["payment_id"]),
            "starts_at": str(rows["prediction_time"].min()),
            "ends_at": str(rows["prediction_time"].max()),
        }
    manifest = {
        "dataset": {
            "name": "online_training_features",
            "version": "2.0-online",
            "path": "ml/data/processed/online_training_features.csv",
            "sha256": file_sha256(OUTPUT_DATASET_PATH),
            "rows": len(output),
            "provenance": "SYNTHETIC",
            "source_dataset": "logging_policy_dataset",
            "source_dataset_sha256": file_sha256(SOURCE_DATASET_PATH),
            "real_shadow_cases_included": False,
        },
        "split": {
            "type": "frozen_temporal_membership",
            "column": "split",
            **split_manifest,
        },
        "random_seed": int(source_manifest["random_seed"]),
        "target": {"name": TARGET},
        "identifiers": ["payment_id", "customer_id"],
        "categorical_features": categorical,
        "numerical_features": numerical,
        "evaluation_metadata": EVALUATION_METADATA,
        "forbidden_patterns": source_manifest["forbidden_patterns"],
        "online_model_schema_sha256": file_sha256(ONLINE_SCHEMA_PATH),
        "source_manifest_sha256": file_sha256(SOURCE_MANIFEST_PATH),
    }
    OUTPUT_MANIFEST_PATH.write_text(
        yaml.safe_dump(manifest, sort_keys=False),
        encoding="utf-8",
    )

    disposition = {
        "version": 1,
        "model_version": schema["model_version"],
        "kept_features": features,
        "kept_count": len(features),
        "removed_features": schema["unsupported_features"]["remove"],
        "removed_count": len(schema["unsupported_features"]["remove"]),
        "replaced_features": schema["unsupported_features"]["replace"],
        "replacement_count": len(schema["unsupported_features"]["replace"]),
        "policy": {
            "unsupported_unknown_sentinels_as_model_signal": False,
            "real_shadow_training_allowed": False,
            "failure_taxonomy_is_model_feature": False,
        },
        "dataset": {
            "path": str(OUTPUT_DATASET_PATH.relative_to(REPO_ROOT)).replace(
                "\\",
                "/",
            ),
            "rows": len(output),
            "sha256": manifest["dataset"]["sha256"],
        },
    }
    _write_json(REPORT_PATH, disposition)
    return manifest


if __name__ == "__main__":
    result = build_online_training_dataset()
    print(json.dumps(result, indent=2, sort_keys=True))
