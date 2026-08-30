from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqlalchemy.orm import Session

from app.ml.model_loader import load_model_bundle
from app.services.execution_gate import load_execution_gate
from app.services.shadow_monitoring import shadow_metrics


REPO_ROOT = Path(__file__).resolve().parents[3]
PARITY_REPORT_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase9" / "feature_contract_report.json"
)


def evaluate_shadow_gate(session: Session) -> dict[str, Any]:
    config = load_execution_gate()
    thresholds = config["shadow_evaluation"]
    metrics = shadow_metrics(session)
    parity_report = json.loads(PARITY_REPORT_PATH.read_text(encoding="utf-8"))
    artifact_valid = False
    try:
        load_model_bundle()
        artifact_valid = True
    except Exception:
        artifact_valid = False

    pipeline_cases = int(metrics["cases"])
    cases = int(metrics["shadow_decisions"])
    failed = int(metrics["failed_inferences"])
    successful = int(metrics["successful_inferences"])
    terminal_jobs = successful + failed
    failed_rate = failed / terminal_jobs if terminal_jobs else 0.0
    worker_reliable = (
        pipeline_cases > 0
        and successful == pipeline_cases
        and failed == 0
        and not any(
            count
            for status, count in metrics["job_statuses"].items()
            if status not in {"succeeded"}
        )
    )
    unknown_feature_rate = float(
        metrics.get("feature_drift", {}).get("unknown_category_rate", 1.0)
    )
    unknown_failure_rate = float(
        metrics["unknown_failure_rate"]
        if metrics["unknown_failure_rate"] is not None
        else 1.0
    )
    checks = {
        "minimum_real_cases": {
            "passed": cases >= int(thresholds["minimum_real_cases"]),
            "observed": cases,
            "required": int(thresholds["minimum_real_cases"]),
        },
        "inference_failure_rate": {
            "passed": failed_rate
            <= float(thresholds["maximum_failed_inference_rate"]),
            "observed": failed_rate,
            "maximum": float(
                thresholds["maximum_failed_inference_rate"]
            ),
        },
        "unknown_failure_rate": {
            "passed": unknown_failure_rate
            <= float(thresholds["maximum_unknown_failure_rate"]),
            "observed": unknown_failure_rate,
            "maximum": float(thresholds["maximum_unknown_failure_rate"]),
        },
        "unknown_feature_rate": {
            "passed": unknown_feature_rate
            <= float(thresholds["maximum_unknown_feature_rate"]),
            "observed": unknown_feature_rate,
            "maximum": float(thresholds["maximum_unknown_feature_rate"]),
        },
        "policy_violations": {
            "passed": int(metrics["policy_violations"])
            <= int(thresholds["maximum_policy_violations"]),
            "observed": int(metrics["policy_violations"]),
            "maximum": int(thresholds["maximum_policy_violations"]),
        },
        "automated_fraud_actions": {
            "passed": int(metrics["automated_fraud_actions"])
            <= int(thresholds["maximum_automated_fraud_actions"]),
            "observed": int(metrics["automated_fraud_actions"]),
            "maximum": int(
                thresholds["maximum_automated_fraud_actions"]
            ),
        },
        "duplicate_decisions": {
            "passed": int(metrics["duplicate_decisions"])
            <= int(thresholds["maximum_duplicate_decisions"]),
            "observed": int(metrics["duplicate_decisions"]),
            "maximum": int(thresholds["maximum_duplicate_decisions"]),
        },
        "feature_parity": {
            "passed": parity_report["parity_verification"]["status"]
            == "passed",
            "observed": parity_report["parity_verification"]["status"],
            "required": "passed",
        },
        "artifact_validation": {
            "passed": artifact_valid,
            "observed": artifact_valid,
            "required": True,
        },
        "worker_reliability": {
            "passed": worker_reliable,
            "observed": worker_reliable,
            "required": True,
        },
    }
    ready = all(check["passed"] for check in checks.values())
    return {
        "status": "ready" if ready else "blocked",
        "execution_mode": config["execution"]["mode"],
        "controlled_execution_authorized": bool(
            config["execution"]["controlled_execution_authorized"]
        ),
        "checks": checks,
        "metrics": metrics,
    }
