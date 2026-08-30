from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "final"
BASELINE_PATH = REPORT_DIR / "baseline_manifest.json"

BASELINE_PATHS: dict[str, str] = {
    "v1_model": "ml/artifacts/recovery_model_v1.json",
    "v1_preprocessor": "ml/artifacts/preprocessing_v1.joblib",
    "v1_calibrator": "ml/artifacts/calibration_v1.joblib",
    "v1_metadata": "ml/artifacts/model_metadata.json",
    "v2_model": "ml/artifacts/v2_online/recovery_model_v2_online.json",
    "v2_preprocessor": "ml/artifacts/v2_online/preprocessing_v2_online.joblib",
    "v2_calibrator": "ml/artifacts/v2_online/calibration_v2_online.joblib",
    "v2_manifest": "ml/config/recovery_model_v2_online_manifest.yaml",
    "feature_schema": "ml/config/feature_schema.yaml",
    "online_model_schema": "ml/config/online_model_schema.yaml",
    "policy_v3_manifest": "ml/config/policy_v3_manifest.yaml",
    "policy_evaluation": "ml/config/policy_evaluation.yaml",
    "execution_gate": "ml/config/execution_gate.yaml",
    "intervention_policy": "ml/config/intervention_policy.yaml",
    "action_matrix": "ml/config/action_matrix.yaml",
    "outcome_observation": "ml/config/outcome_observation.yaml",
    "real_outcome_schema": "ml/config/real_outcome_schema.yaml",
    "controlled_pilot": "ml/config/controlled_pilot.yaml",
    "qwen_agent": "ml/config/qwen_agent.yaml",
    "phase11_baseline": "ml/config/phase11_baseline_manifest.yaml",
    "phase11_shadow_validation": "ml/reports/phase11/phase11_shadow_validation.json",
    "phase11_cases": "ml/reports/phase11/real_shadow_cases.csv",
    "phase12_manifest": "ml/reports/phase12/phase12_manifest.json",
    "phase12_readiness": "ml/reports/phase12/phase12_readiness.json",
    "phase13_report_manifest": "ml/reports/phase13/phase13_report_manifest.json",
    "phase13_readiness": "ml/reports/phase13/phase13_readiness.json",
    "phase14_manifest": "ml/reports/phase14/phase14_manifest.json",
    "phase14_summary": "ml/reports/phase14/phase14_summary.json",
    "phase14_coverage": "ml/reports/phase14/coverage.json",
    "phase15_readiness": "ml/reports/phase15/phase15_readiness.json",
}


def sha256_file(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_baseline_manifest() -> dict[str, Any]:
    artifacts: dict[str, Any] = {}
    missing: list[str] = []
    for key, relative in BASELINE_PATHS.items():
        path = REPO_ROOT / relative
        digest = sha256_file(path)
        artifacts[key] = {
            "path": relative.replace("\\", "/"),
            "sha256": digest,
            "exists": digest is not None,
        }
        if digest is None:
            missing.append(relative)
    return {
        "version": 1,
        "phase": "final",
        "generated_at": datetime.now(UTC).isoformat(),
        "execution_mode": "shadow",
        "controlled_execution_authorized": False,
        "phase15_authorized": False,
        "model_ready": False,
        "qwen_tools_enabled": False,
        "policy_version": "recovery_policy_v3",
        "v1": {
            "version": "recovery_model_v1",
            "model_hash": artifacts["v1_model"]["sha256"],
            "preprocessor_hash": artifacts["v1_preprocessor"]["sha256"],
            "calibrator_hash": artifacts["v1_calibrator"]["sha256"],
        },
        "v2_online": {
            "version": "recovery_model_v2_online",
            "model_hash": artifacts["v2_model"]["sha256"],
            "preprocessor_hash": artifacts["v2_preprocessor"]["sha256"],
            "calibrator_hash": artifacts["v2_calibrator"]["sha256"],
        },
        "feature_schema": artifacts["feature_schema"]["sha256"],
        "artifacts": artifacts,
        "missing_artifacts": missing,
        "freeze_note": (
            "Final baseline freeze. Do not mutate hashed upstream artifacts "
            "during the final phase."
        ),
    }


def write_baseline_manifest(path: Path = BASELINE_PATH) -> dict[str, Any]:
    payload = build_baseline_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> None:
    payload = write_baseline_manifest()
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
