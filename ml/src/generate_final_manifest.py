from __future__ import annotations

"""
Generate ml/config/final_manifest.yaml — frozen hashes for the final phase.

Does not retrain or mutate artifacts. Re-run after intentional freezes only.
"""

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = REPO_ROOT / "ml" / "config" / "final_manifest.yaml"

ARTIFACTS: dict[str, str] = {
    "v1_model": "ml/artifacts/recovery_model_v1.json",
    "v1_preprocessor": "ml/artifacts/preprocessing_v1.joblib",
    "v1_calibrator": "ml/artifacts/calibration_v1.joblib",
    "v1_metadata": "ml/artifacts/model_metadata.json",
    "v2_model": "ml/artifacts/v2_online/recovery_model_v2_online.json",
    "v2_preprocessor": "ml/artifacts/v2_online/preprocessing_v2_online.joblib",
    "v2_calibrator": "ml/artifacts/v2_online/calibration_v2_online.joblib",
    "feature_schema": "ml/config/feature_schema.yaml",
    "online_model_schema": "ml/config/online_model_schema.yaml",
    "online_feature_schema": "ml/config/online_feature_schema.yaml",
    "policy_v3_manifest": "ml/config/policy_v3_manifest.yaml",
    "intervention_policy": "ml/config/intervention_policy.yaml",
    "action_matrix": "ml/config/action_matrix.yaml",
    "execution_gate": "ml/config/execution_gate.yaml",
    "controlled_pilot": "ml/config/controlled_pilot.yaml",
    "agent_actions": "ml/config/agent_actions.yaml",
    "qwen_agent": "ml/config/qwen_agent.yaml",
    "qwen": "ml/config/qwen.yaml",
    "failure_taxonomy": "ml/config/razorpay_failure_taxonomy.yaml",
    "outcome_observation": "ml/config/outcome_observation.yaml",
    "real_outcome_schema": "ml/config/real_outcome_schema.yaml",
    "model_readiness": "ml/config/model_readiness.yaml",
    "training_dataset_v1": "ml/data/processed/logging_policy_dataset.csv",
    "training_dataset_v2_online": "ml/data/processed/online_training_dataset.csv",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_final_manifest() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    missing: list[str] = []
    for key, relative in ARTIFACTS.items():
        digest = sha256_file(REPO_ROOT / relative)
        artifacts[key] = {
            "path": relative.replace("\\", "/"),
            "sha256": digest,
            "exists": digest is not None,
        }
        if digest is None:
            missing.append(relative)
    return {
        "version": 1,
        "schema_id": "recoverai_final_manifest_v1",
        "generated_at": datetime.now(UTC).isoformat(),
        "phase": "final",
        "freeze": {
            "xgboost_v1": True,
            "xgboost_v2_online": True,
            "feature_schema": True,
            "policy_v3": True,
            "failure_taxonomy": True,
            "safety_rules": True,
            "recovery_agent": True,
            "qwen_validation": True,
            "outcome_observer": True,
        },
        "authority": {
            "xgboost": "recovery_probability",
            "policy": "intervention_selection",
            "safety": "permit_or_block",
            "agent": "communication",
            "qwen": "language_generation",
            "executor": "permitted_action_only",
        },
        "execution": {
            "mode": "shadow",
            "controlled_execution_authorized": False,
            "pilot_enabled": False,
            "provider_actions_enabled": False,
            "qwen_tools_enabled": False,
            "first_controlled_action": "payment_link",
        },
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "note": (
            "Frozen final-phase inventory. Do not retrain or flip gates "
            "until legitimate attributed recoveries open the evidence gate."
        ),
    }


def write_final_manifest(path: Path = MANIFEST_PATH) -> dict[str, Any]:
    payload = build_final_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = write_final_manifest()
    print(f"Wrote {MANIFEST_PATH}")
    print(f"missing={payload['missing_artifacts']}")


if __name__ == "__main__":
    main()
