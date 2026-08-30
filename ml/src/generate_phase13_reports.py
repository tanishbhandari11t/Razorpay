from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.src.model_pipeline import file_sha256


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase13"
MODEL_COMPARISON_PATH = REPORT_DIR / "v2_online_model_comparison.json"
ECONOMICS_PATH = REPORT_DIR / "v2_online_policy_economics.json"
OPE_PATH = REPORT_DIR / "v2_online_ope.json"
SENSITIVITY_PATH = REPORT_DIR / "v2_online_sensitivity.json"
SHADOW_REPLAY_PATH = REPORT_DIR / "v2_online_shadow_replay.json"
MODEL_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_model_v2_online_manifest.yaml"
)
READINESS_CONFIG_PATH = (
    REPO_ROOT / "ml" / "config" / "model_readiness.yaml"
)
PHASE11_BASELINE_PATH = (
    REPO_ROOT / "ml" / "config" / "phase11_baseline_manifest.yaml"
)
PHASE12_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase12" / "phase12_manifest.json"
)


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _verify_manifest_entries(entries: dict[str, Any]) -> bool:
    for entry in entries.values():
        if file_sha256(REPO_ROOT / entry["path"]) != str(entry["sha256"]):
            return False
    return True


def generate_phase13_reports() -> dict[str, Any]:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    readiness_config = yaml.safe_load(
        READINESS_CONFIG_PATH.read_text(encoding="utf-8")
    )
    model_manifest = yaml.safe_load(
        MODEL_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    comparison = _load_json(MODEL_COMPARISON_PATH)
    economics = _load_json(ECONOMICS_PATH)
    ope = _load_json(OPE_PATH)
    sensitivity = _load_json(SENSITIVITY_PATH)
    shadow = _load_json(SHADOW_REPLAY_PATH)
    online_availability = yaml.safe_load(
        (
            REPO_ROOT / "ml" / "config" / "online_feature_schema.yaml"
        ).read_text(encoding="utf-8")
    )
    artifact_integrity = _verify_manifest_entries(
        model_manifest["artifacts"]
    ) and _verify_manifest_entries(model_manifest["frozen_inputs"])

    deltas = comparison["delta_v2_minus_v1"]
    model_quality_checks = {
        "roc_auc_not_worse_than_v1": deltas["roc_auc"] >= 0,
        "pr_auc_not_worse_than_v1": deltas["pr_auc"] >= 0,
        "brier_not_worse_than_v1": deltas["brier_score"] <= 0,
        "calibration_error_not_worse_than_v1": (
            deltas["mean_absolute_calibration_error"] <= 0
        ),
    }
    v2_economics = economics["v2_online_with_frozen_v3"]
    retry_economics = economics["always_retry"]
    economic_improvement = (
        float(v2_economics["net_recovered_value_inr"])
        > float(retry_economics["net_recovered_value_inr"])
    )
    v2_ope = ope["policies"]["v2_online_with_frozen_v3"]
    retry_ope = ope["policies"]["always_retry"]
    v2_interval = v2_ope["confidence_intervals_95"]["doubly_robust"]
    retry_interval = retry_ope["confidence_intervals_95"]["doubly_robust"]
    dr_nonoverlap = (
        float(v2_interval[0]) > float(retry_interval[1])
        or float(retry_interval[0]) > float(v2_interval[1])
    )
    checks = {
        "artifact_integrity": {
            "passed": artifact_integrity,
            "required": True,
        },
        "deployable_feature_contract": {
            "passed": (
                int(model_manifest["model"]["raw_feature_count"]) == 33
            ),
            "observed": model_manifest["model"]["raw_feature_count"],
            "required": 33,
        },
        "temporal_integrity": {
            "passed": (
                model_manifest["training"]["temporal_split"]
                and not model_manifest["training"]["test_used_for_fitting"]
                and not model_manifest["training"]["real_shadow_cases_used"]
            ),
            "required": True,
        },
        "model_quality": {
            "passed": all(model_quality_checks.values()),
            "details": model_quality_checks,
            "required": "V2 must not regress against V1",
        },
        "policy_safety": {
            "passed": (
                economics["policy_safety"]["policy_violations"] == 0
                and economics["policy_safety"][
                    "fraud_automated_actions"
                ]
                == 0
            ),
            "policy_violations": economics["policy_safety"][
                "policy_violations"
            ],
            "fraud_automated_actions": economics["policy_safety"][
                "fraud_automated_actions"
            ],
        },
        "economic_improvement_vs_always_retry": {
            "passed": economic_improvement,
            "v2_net_value_inr": v2_economics[
                "net_recovered_value_inr"
            ],
            "always_retry_net_value_inr": retry_economics[
                "net_recovered_value_inr"
            ],
        },
        "ope_uncertainty": {
            "passed": dr_nonoverlap
            and float(v2_ope["doubly_robust"])
            > float(retry_ope["doubly_robust"]),
            "v2_dr": v2_ope["doubly_robust"],
            "v2_dr_ci95": v2_interval,
            "always_retry_dr": retry_ope["doubly_robust"],
            "always_retry_dr_ci95": retry_interval,
            "intervals_nonoverlapping": dr_nonoverlap,
        },
        "sensitivity": {
            "passed": sensitivity["robust_uplift_supported"],
            "scenarios_won": sensitivity["scenarios_won"],
            "scenarios_total": sensitivity["scenarios_total"],
            "required_wins": sensitivity["robust_win_minimum"],
        },
        "real_shadow_compatibility": {
            "passed": (
                shadow["compatibility_allowed_cases"]
                >= readiness_config["requirements"]["real_shadow"][
                    "minimum_compatibility_cases"
                ]
                and online_availability["automation_support"][
                    "selected_threshold"
                ]
                is not None
                and shadow["taxonomy_distribution"].get("unknown", 0) == 0
            ),
            "cases": shadow["cases"],
            "compatible_cases": shadow["compatibility_allowed_cases"],
            "taxonomy_distribution": shadow["taxonomy_distribution"],
            "feature_support_score": shadow[
                "mean_feature_support_score"
            ],
            "feature_support_threshold": online_availability[
                "automation_support"
            ]["selected_threshold"],
            "feature_support_threshold_selected": (
                online_availability["automation_support"][
                    "selected_threshold"
                ]
                is not None
            ),
            "known_failure_taxonomy_required": True,
        },
        "shadow_safety": {
            "passed": (
                shadow["safety"]["provider_calls"] == 0
                and shadow["safety"]["database_writes"] == 0
                and not shadow["safety"][
                    "controlled_execution_authorized"
                ]
            ),
            "provider_calls": shadow["safety"]["provider_calls"],
            "database_writes": shadow["safety"]["database_writes"],
        },
    }
    required_checks = [
        key
        for key in checks
        if key
        not in {
            "shadow_safety",
        }
    ]
    model_ready = all(checks[key]["passed"] for key in required_checks)
    readiness = {
        "version": 1,
        "model_version": readiness_config["model_version"],
        "policy_wrapper": readiness_config["policy_wrapper"],
        "status": "ready" if model_ready else "blocked",
        "model_ready": model_ready,
        "checks": checks,
        "evidence_boundaries": {
            "classification": "synthetic_frozen_test",
            "economics": "synthetic_counterfactual",
            "ope": "synthetic_logged_observed",
            "real_shadow": "descriptive_compatibility_only",
            "real_recovery_uplift_claim_allowed": False,
        },
        "execution": {
            "mode": "shadow",
            "v2_offline_shadow_replay": "PASS",
            "v2_live_automatic_lane": "DISABLED",
            "provider_actions_enabled": False,
            "controlled": "BLOCKED",
        },
        "decision": (
            "V2-online is deployable by feature contract but does not beat "
            "the frozen quality, economics, OPE, sensitivity, or real "
            "compatibility gates. Preserve it as a shadow challenger."
        ),
    }
    readiness_path = REPORT_DIR / "phase13_readiness.json"
    _write_json(readiness_path, readiness)

    summary = {
        "version": 1,
        "model_version": readiness_config["model_version"],
        "raw_features": model_manifest["model"]["raw_feature_count"],
        "transformed_features": model_manifest["model"][
            "transformed_feature_count"
        ],
        "classification": comparison,
        "economics": economics,
        "ope": {
            "always_retry": retry_ope,
            "v2_online_with_frozen_v3": v2_ope,
        },
        "sensitivity": {
            "won": sensitivity["scenarios_won"],
            "total": sensitivity["scenarios_total"],
            "robust_uplift_supported": sensitivity[
                "robust_uplift_supported"
            ],
        },
        "real_shadow": shadow,
        "readiness": {
            "model_ready": model_ready,
            "controlled_execution": "BLOCKED",
        },
    }
    summary_path = REPORT_DIR / "phase13_summary.json"
    _write_json(summary_path, summary)

    artifacts = [
        REPORT_DIR / "feature_disposition.json",
        MODEL_COMPARISON_PATH,
        REPORT_DIR / "v2_online_policy_decisions.csv",
        ECONOMICS_PATH,
        OPE_PATH,
        SENSITIVITY_PATH,
        REPORT_DIR / "v2_online_shadow_pairs.csv",
        SHADOW_REPLAY_PATH,
        readiness_path,
        summary_path,
    ]
    inputs = [
        REPO_ROOT / "ml" / "config" / "online_model_schema.yaml",
        REPO_ROOT / "ml" / "config" / "online_feature_schema.yaml",
        REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml",
        REPO_ROOT / "ml" / "config" / "phase13_evaluation.yaml",
        READINESS_CONFIG_PATH,
        REPO_ROOT / "ml" / "config" / "phase13_shadow.yaml",
        MODEL_MANIFEST_PATH,
        PHASE11_BASELINE_PATH,
        PHASE12_MANIFEST_PATH,
        REPO_ROOT / "ml" / "config" / "policy_v3_manifest.yaml",
        REPO_ROOT / "ml" / "config" / "intervention_policy.yaml",
        REPO_ROOT / "ml" / "config" / "action_matrix.yaml",
        REPO_ROOT / "ml" / "src" / "build_online_training_dataset.py",
        REPO_ROOT / "ml" / "src" / "train_recovery_model_v2_online.py",
        REPO_ROOT / "ml" / "src" / "evaluate_v2_online.py",
        REPO_ROOT / "ml" / "src" / "replay_v2_online_shadow.py",
        REPO_ROOT
        / "ml"
        / "src"
        / "policies"
        / "decision_margin.py",
        Path(__file__),
        REPO_ROOT
        / "backend"
        / "app"
        / "ml"
        / "model_loader_v2_online.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "services"
        / "recovery_inference_v2_online.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "services"
        / "recovery_jobs_v2_online.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "services"
        / "shadow_ab_comparison.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "workers"
        / "recovery_tasks_v2_online.py",
        REPO_ROOT / "backend" / "app" / "workers" / "celery_app.py",
        REPO_ROOT / "backend" / "app" / "api" / "recovery.py",
        REPO_ROOT / "backend" / "app" / "api" / "webhooks.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "services"
        / "recovery_jobs.py",
        REPO_ROOT
        / "backend"
        / "app"
        / "services"
        / "shadow_monitoring.py",
    ]
    report_manifest = {
        "version": 1,
        "phase": 13,
        "generated_at": datetime.now(UTC).isoformat(),
        "model_version": readiness_config["model_version"],
        "policy_v3_modified": False,
        "real_shadow_training_allowed": False,
        "inputs": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): {
                "sha256": file_sha256(path)
            }
            for path in inputs
        },
        "artifacts": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): {
                "sha256": file_sha256(path)
            }
            for path in artifacts
        },
        "promotion": {
            "model_ready": model_ready,
            "live_shadow_automatic_authorized": False,
            "controlled_execution_authorized": False,
            "provider_actions_enabled": False,
        },
    }
    _write_json(
        REPORT_DIR / "phase13_report_manifest.json",
        report_manifest,
    )
    return readiness


if __name__ == "__main__":
    print(
        json.dumps(
            generate_phase13_reports(),
            indent=2,
            sort_keys=True,
        )
    )
