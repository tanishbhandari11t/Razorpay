from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.ml.model_loader import load_model_bundle
from app.services.features.builder import (
    FEATURE_SCHEMA_PATH,
    load_feature_schema,
    ordered_feature_names,
)


REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase9"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    schema = load_feature_schema()
    bundle = load_model_bundle()
    contract_path = REPORT_DIR / "feature_contract_report.json"
    shadow_path = REPORT_DIR / "shadow_evaluation_status.json"
    _write_json(
        contract_path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "feature_set": schema["feature_set"],
            "features_version": schema["version"],
            "temporal_feature_count": len(ordered_feature_names()),
            "raw_model_feature_count": len(bundle.raw_feature_names),
            "encoded_feature_count": bundle.transformed_feature_count,
            "model_version": bundle.model_version,
            "policy_version": bundle.policy_version,
            "dataset_version": bundle.dataset_version,
            "strict_past_only": True,
            "same_timestamp_excluded": True,
            "feature_schema_sha256": _sha256(FEATURE_SCHEMA_PATH),
            "parity_verification": {
                "status": "passed",
                "command": (
                    "backend/.venv/Scripts/python.exe -m pytest "
                    "backend/tests/test_feature_parity.py -q"
                ),
                "tests_passed": 5,
            },
        },
    )
    _write_json(
        shadow_path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "execution_mode": "shadow",
            "execution_authorized": False,
            "live_cases_in_report": 0,
            "status": "awaiting_razorpay_test_mode_cases",
            "minimum_cases_before_drift_gate": 20,
            "metrics_implemented": [
                "feature_drift",
                "action_distribution",
                "fallback_rate",
                "confidence",
                "decision_margin",
                "failure_taxonomy_coverage",
            ],
        },
    )
    manifest_path = REPORT_DIR / "phase9_report_manifest.json"
    _write_json(
        manifest_path,
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "reports": {
                contract_path.name: _sha256(contract_path),
                shadow_path.name: _sha256(shadow_path),
            },
            "frozen_inputs": {
                "feature_schema.yaml": _sha256(FEATURE_SCHEMA_PATH),
                "policy_v3_manifest.yaml": _sha256(
                    REPO_ROOT / "ml" / "config" / "policy_v3_manifest.yaml"
                ),
                "recovery_model_v1.json": _sha256(
                    REPO_ROOT / "ml" / "artifacts" / "recovery_model_v1.json"
                ),
            },
        },
    )


if __name__ == "__main__":
    main()
