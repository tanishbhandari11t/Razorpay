from __future__ import annotations

import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from ml.src.failure_classifier import load_intervention_policy
from ml.src.feature_support import (
    automation_eligibility,
    evaluate_feature_support,
)
from ml.src.model_pipeline import (
    feature_columns,
    load_manifest,
    predict_probabilities,
)
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    decide_recovery_action,
    load_action_matrix,
)
from ml.src.policies.decision_margin import apply_decision_margin
from ml.src.policies.stopping_rules import RecoveryPolicyContext
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    load_support_policy_config,
)
from ml.src.razorpay_failure_taxonomy import (
    classify_razorpay_failure,
    legacy_failure_class,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
FROZEN_CASES_PATH = (
    REPO_ROOT
    / "ml"
    / "reports"
    / "phase12"
    / "real_razorpay_shadow_v1_features.json"
)
V1_MANIFEST_PATH = REPO_ROOT / "ml" / "config" / "dataset_manifest.yaml"
V2_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
V1_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts"
V2_ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "v2_online"
CONTEXT_SUPPORT_PATH = REPO_ROOT / "ml" / "reports" / "context_support.csv"
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase13"
EVALUATION_CONFIG_PATH = (
    REPO_ROOT / "ml" / "config" / "phase13_evaluation.yaml"
)


def _load_bundle(
    artifact_dir: Path,
    version: str,
) -> tuple[Any, XGBClassifier, Any]:
    preprocessor = joblib.load(
        artifact_dir / f"preprocessing_{version}.joblib"
    )
    calibrator = joblib.load(
        artifact_dir / f"calibration_{version}.joblib"
    )
    model = XGBClassifier()
    model.load_model(artifact_dir / f"recovery_model_{version}.json")
    return preprocessor, model, calibrator


def _candidate_probabilities(
    features: dict[str, Any],
    manifest: dict[str, Any],
    artifact_dir: Path,
    version: str,
    actions: list[str],
) -> dict[str, float]:
    preprocessor, model, calibrator = _load_bundle(
        artifact_dir,
        version,
    )
    rows = []
    columns = feature_columns(manifest)
    for action in actions:
        candidate = dict(features)
        candidate["chosen_intervention"] = action
        rows.append({name: candidate[name] for name in columns})
    probabilities = predict_probabilities(
        preprocessor,
        model,
        pd.DataFrame(rows, columns=columns),
        calibrator,
    )
    return {
        action: float(probability)
        for action, probability in zip(actions, probabilities, strict=True)
    }


def _margin(probabilities: dict[str, float]) -> float:
    values = sorted(probabilities.values(), reverse=True)
    return values[0] - values[1]


def replay_shadow() -> dict[str, Any]:
    frozen = json.loads(FROZEN_CASES_PATH.read_text(encoding="utf-8"))
    if frozen["dataset_id"] != "REAL_RAZORPAY_SHADOW_V1":
        raise ValueError("Unexpected real shadow dataset")
    if frozen["training_allowed"] is not False:
        raise ValueError("Real shadow cases must never be training data")
    if frozen["case_count"] != 20:
        raise ValueError("Frozen real shadow set must contain 20 cases")
    v1_manifest = load_manifest(V1_MANIFEST_PATH)
    v2_manifest = load_manifest(V2_MANIFEST_PATH)
    support_config = load_support_policy_config()
    support_index = SupportIndex(
        pd.read_csv(CONTEXT_SUPPORT_PATH),
        support_config,
    )
    policy = load_intervention_policy()
    matrix = load_action_matrix()
    evaluation = yaml.safe_load(
        EVALUATION_CONFIG_PATH.read_text(encoding="utf-8")
    )
    actions = [str(action) for action in support_config["interventions"]]
    rows = []
    for case in frozen["cases"]:
        features = case["features"]
        taxonomy = classify_razorpay_failure(
            case["raw_failure_reason"],
            fraud_flag=int(features["fraud_flag"]),
        )
        support_result = evaluate_feature_support(features)
        compatibility = automation_eligibility(
            failure_taxonomy=taxonomy.taxonomy,
            support=support_result,
        )
        prediction_time = datetime.fromisoformat(features["prediction_time"])
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=UTC)
        context = RecoveryPolicyContext(
            case_id=str(case["case_id"]),
            payment_id=str(case["payment_id"]),
            amount_inr=float(features["amount_inr"]),
            payment_status="failed",
            failure_reason=legacy_failure_class(taxonomy),
            fraud_flag=int(features["fraud_flag"]),
            case_created_at=prediction_time,
            now=prediction_time,
            customer_contact_available=True,
            customer_opted_out=False,
            valid_payment_context=True,
        )
        candidate_support = {}
        for action in actions:
            evidence = support_index.evidence(features, action)
            candidate_support[action] = CandidateSupport(
                supported=evidence.supported,
                action_count=evidence.action_count,
                effective_sample_size=evidence.effective_sample_size,
            )
        v1_probabilities = _candidate_probabilities(
            features,
            v1_manifest,
            V1_ARTIFACT_DIR,
            "v1",
            actions,
        )
        v2_probabilities = _candidate_probabilities(
            features,
            v2_manifest,
            V2_ARTIFACT_DIR,
            "v2_online",
            actions,
        )
        v1_decision = decide_recovery_action(
            context,
            v1_probabilities,
            candidate_support,
            policy=policy,
            action_matrix=matrix,
        )
        v2_decision = decide_recovery_action(
            context,
            v2_probabilities,
            candidate_support,
            policy=policy,
            action_matrix=matrix,
        )
        v2_margin_gate = apply_decision_margin(
            v2_decision,
            fallback_action=str(
                support_config["decision"]["preferred_fallback_action"]
            ),
            threshold_inr=float(
                evaluation["decision_margins"][
                    "minimum_expected_value_margin_inr"
                ]
            ),
        )
        effective_v1 = (
            "escalate_to_merchant"
            if not compatibility["allowed"]
            else v1_decision.selected_action or "no_action"
        )
        effective_v2 = (
            "escalate_to_merchant"
            if not compatibility["allowed"]
            else v2_margin_gate.selected_action or "no_action"
        )
        rows.append(
            {
                "payment_id": case["payment_id"],
                "raw_failure_reason": case["raw_failure_reason"],
                "taxonomy": taxonomy.taxonomy,
                "feature_support_score": support_result.score,
                "compatibility_allowed": compatibility["allowed"],
                "v1_pre_gate_action": (
                    v1_decision.selected_action or "no_action"
                ),
                "v2_pre_gate_action": (
                    v2_margin_gate.selected_action or "no_action"
                ),
                "v1_post_gate_action": effective_v1,
                "v2_post_gate_action": effective_v2,
                "post_gate_agreement": effective_v1 == effective_v2,
                "v1_probability_margin": _margin(v1_probabilities),
                "v2_probability_margin": _margin(v2_probabilities),
                "v2_expected_value_margin_inr": (
                    v2_margin_gate.decision_margin_inr
                ),
                "v2_margin_fallback_triggered": (
                    v2_margin_gate.fallback_triggered
                ),
                "v1_probabilities": json.dumps(
                    v1_probabilities,
                    sort_keys=True,
                ),
                "v2_probabilities": json.dumps(
                    v2_probabilities,
                    sort_keys=True,
                ),
                "executed": False,
            }
        )
    pairs = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(
        REPORT_DIR / "v2_online_shadow_pairs.csv",
        index=False,
        float_format="%.8f",
    )
    report = {
        "version": 1,
        "dataset_id": frozen["dataset_id"],
        "cases": len(pairs),
        "evidence_type": "real_shadow_descriptive_replay",
        "training_allowed": False,
        "recovery_uplift_claim_allowed": False,
        "taxonomy_distribution": dict(Counter(pairs["taxonomy"])),
        "mean_feature_support_score": round(
            float(pairs["feature_support_score"].mean()),
            8,
        ),
        "compatibility_allowed_cases": int(
            pairs["compatibility_allowed"].sum()
        ),
        "pre_gate_action_agreement_rate": round(
            float(
                pairs["v1_pre_gate_action"].eq(
                    pairs["v2_pre_gate_action"]
                ).mean()
            ),
            8,
        ),
        "post_gate_action_agreement_rate": round(
            float(pairs["post_gate_agreement"].mean()),
            8,
        ),
        "v1_mean_probability_margin": round(
            float(pairs["v1_probability_margin"].mean()),
            8,
        ),
        "v2_mean_probability_margin": round(
            float(pairs["v2_probability_margin"].mean()),
            8,
        ),
        "v2_margin_fallback_count": int(
            pairs["v2_margin_fallback_triggered"].sum()
        ),
        "post_gate_action_distribution": {
            "v1": dict(Counter(pairs["v1_post_gate_action"])),
            "v2_online": dict(Counter(pairs["v2_post_gate_action"])),
        },
        "safety": {
            "unknown_failure_action": "escalate_to_merchant",
            "provider_calls": 0,
            "database_writes": 0,
            "controlled_execution_authorized": False,
        },
    }
    (REPORT_DIR / "v2_online_shadow_replay.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


if __name__ == "__main__":
    print(json.dumps(replay_shadow(), indent=2, sort_keys=True))
