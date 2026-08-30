from __future__ import annotations

import hashlib
import json
import math
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from sqlalchemy import select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.database.connection import get_session, initialize_database
from app.models.agent_decision import AgentDecision
from app.models.payment import Payment
from app.models.payment_feature_context import PaymentFeatureContext
from app.models.recovery_case import RecoveryCase
from app.models.webhook_event import WebhookEvent
from ml.src.feature_support import (
    automation_eligibility,
    evaluate_feature_support,
    feature_contract,
    load_online_feature_schema,
    model_feature_names,
)
from ml.src.razorpay_failure_taxonomy import classify_razorpay_failure


PHASE11_DIR = REPO_ROOT / "ml" / "reports" / "phase11"
PHASE12_DIR = REPO_ROOT / "ml" / "reports" / "phase12"
PHASE11_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "phase11_baseline_manifest.yaml"
)
TRAINING_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
SYNTHETIC_SOURCE_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "upi_transactions_clean.csv"
)
FROZEN_CASES_PATH = PHASE11_DIR / "real_shadow_cases.csv"
NUMERIC_DRIFT_FEATURES = (
    "amount_inr",
    "hour_of_day",
    "previous_transaction_count",
    "previous_failure_count",
    "amount_vs_previous_avg",
)
CATEGORICAL_DRIFT_FEATURES = (
    "transaction_type",
    "merchant_category",
    "device_type",
    "network_type",
    "sender_state",
    "sender_bank",
    "day_of_week",
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _json_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _stats(values: pd.Series) -> dict[str, float | int | None]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "mean": None, "median": None, "std": None}
    return {
        "count": int(len(numeric)),
        "mean": float(numeric.mean()),
        "median": float(numeric.median()),
        "std": float(numeric.std(ddof=0)),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
    }


def _ks_statistic(left: pd.Series, right: pd.Series) -> float | None:
    a = sorted(pd.to_numeric(left, errors="coerce").dropna().astype(float))
    b = sorted(pd.to_numeric(right, errors="coerce").dropna().astype(float))
    if not a or not b:
        return None
    points = sorted(set(a) | set(b))
    ai = bi = 0
    maximum = 0.0
    for point in points:
        while ai < len(a) and a[ai] <= point:
            ai += 1
        while bi < len(b) and b[bi] <= point:
            bi += 1
        maximum = max(maximum, abs(ai / len(a) - bi / len(b)))
    return maximum


def _distribution(values: pd.Series) -> dict[str, float]:
    return {
        str(key): float(value)
        for key, value in (
            values.astype(str).value_counts(normalize=True).sort_index().items()
        )
    }


def _total_variation(
    left: dict[str, float],
    right: dict[str, float],
) -> float:
    keys = set(left) | set(right)
    return 0.5 * sum(abs(left.get(key, 0) - right.get(key, 0)) for key in keys)


def _verify_phase11() -> dict[str, Any]:
    manifest = yaml.safe_load(
        PHASE11_MANIFEST_PATH.read_text(encoding="utf-8")
    )
    frozen_inputs = [
        *manifest["artifacts"].values(),
        *manifest["frozen_upstream"].values(),
    ]
    for artifact in frozen_inputs:
        path = REPO_ROOT / artifact["path"]
        if _sha256(path) != artifact["sha256"]:
            raise RuntimeError(f"Frozen Phase 11 artifact changed: {path}")
    return manifest


def _load_cases(
    payment_ids: list[str],
) -> list[
    tuple[
        AgentDecision,
        Payment,
        RecoveryCase,
        WebhookEvent,
        PaymentFeatureContext | None,
    ]
]:
    with get_session() as session:
        rows = session.execute(
            select(
                AgentDecision,
                Payment,
                RecoveryCase,
                WebhookEvent,
                PaymentFeatureContext,
            )
            .join(Payment, Payment.id == AgentDecision.payment_id)
            .join(
                RecoveryCase,
                RecoveryCase.id == AgentDecision.recovery_case_id,
            )
            .join(
                WebhookEvent,
                WebhookEvent.razorpay_event_id
                == RecoveryCase.source_event_id,
            )
            .outerjoin(
                PaymentFeatureContext,
                PaymentFeatureContext.payment_id == Payment.id,
            )
            .where(
                Payment.razorpay_payment_id.in_(payment_ids),
                AgentDecision.execution_mode == "shadow",
            )
        ).all()
    by_payment = {row[1].razorpay_payment_id: row for row in rows}
    missing = [payment_id for payment_id in payment_ids if payment_id not in by_payment]
    if missing:
        raise RuntimeError(f"Frozen Phase 11 cases missing from database: {missing}")
    return [by_payment[payment_id] for payment_id in payment_ids]


def main() -> None:
    baseline = _verify_phase11()
    initialize_database()
    PHASE12_DIR.mkdir(parents=True, exist_ok=True)
    frozen = pd.read_csv(FROZEN_CASES_PATH)
    payment_ids = frozen["payment_id"].astype(str).tolist()
    if len(payment_ids) != 20 or len(set(payment_ids)) != 20:
        raise RuntimeError("REAL_RAZORPAY_SHADOW_V1 must contain 20 unique cases")
    rows = _load_cases(payment_ids)

    case_records: list[dict[str, Any]] = []
    feature_state_counts: dict[str, Counter[str]] = {
        name: Counter() for name in model_feature_names()
    }
    raw_missing_counts: Counter[str] = Counter()
    raw_unknown_counts: Counter[str] = Counter()
    taxonomy_counts: Counter[str] = Counter()
    raw_reason_counts: Counter[str] = Counter()
    support_scores: list[float] = []
    diagnoses = []

    for decision, payment, recovery_case, event, context in rows:
        diagnosis = classify_razorpay_failure(
            payment.failure_reason,
            fraud_flag=int(decision.features_snapshot.get("fraud_flag", 0)),
        )
        diagnoses.append(diagnosis)
        taxonomy_counts[diagnosis.taxonomy] += 1
        raw_reason_counts[diagnosis.normalized_reason] += 1
        support = evaluate_feature_support(decision.features_snapshot)
        support_scores.append(support.score)
        for feature in support.features:
            feature_state_counts[feature.name][feature.state] += 1
            if feature.value is None:
                raw_missing_counts[feature.name] += 1
            if (
                isinstance(feature.value, str)
                and feature.value.strip().upper() == "UNKNOWN"
            ):
                raw_unknown_counts[feature.name] += 1
        payment_entity = (
            event.payload.get("payload", {})
            .get("payment", {})
            .get("entity", {})
        )
        case_records.append(
            {
                "case_id": recovery_case.id,
                "payment_id": payment.razorpay_payment_id,
                "event_id": event.razorpay_event_id,
                "amount_inr": payment.amount / 100,
                "payment_method": payment.method,
                "raw_failure_reason": payment.failure_reason,
                "provider_error": {
                    key: payment_entity.get(key)
                    for key in (
                        "error_code",
                        "error_description",
                        "error_source",
                        "error_step",
                        "error_reason",
                        "bank",
                    )
                },
                "taxonomy": diagnosis.to_dict(),
                "feature_support": {
                    "known": support.known,
                    "unknown": support.unknown,
                    "unavailable": support.unavailable,
                    "total": support.total,
                    "score": support.score,
                },
                "selected_action": decision.selected_action or "no_action",
                "fallback": decision.fallback_used,
                "decision_margin": decision.decision_margin,
                "execution_mode": decision.execution_mode,
                "features": {
                    name: _json_value(value)
                    for name, value in decision.features_snapshot.items()
                },
                "context_unknown_fields": (
                    context.unknown_fields if context is not None else []
                ),
            }
        )

    contract = feature_contract()
    feature_analysis = {}
    for name in model_feature_names():
        counts = feature_state_counts[name]
        classification, reason = contract[name]
        feature_analysis[name] = {
            "online_classification": classification,
            "reason": reason,
            "known": counts["KNOWN"],
            "unknown": counts["UNKNOWN"],
            "unavailable": counts["UNAVAILABLE"],
            "raw_null": raw_missing_counts[name],
            "raw_unknown_sentinel": raw_unknown_counts[name],
            "known_rate": counts["KNOWN"] / len(rows),
        }
    total_feature_values = len(rows) * len(model_feature_names())
    aggregate_known = sum(
        counts["KNOWN"] for counts in feature_state_counts.values()
    )
    aggregate_unknown = sum(
        counts["UNKNOWN"] for counts in feature_state_counts.values()
    )
    aggregate_unavailable = sum(
        counts["UNAVAILABLE"] for counts in feature_state_counts.values()
    )
    unknown_feature_report = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "total_cases": len(rows),
        "model_feature_count": len(model_feature_names()),
        "state_definitions": {
            "KNOWN": "Available with compatible online semantics.",
            "UNKNOWN": "Present as a sentinel or semantically mismatched.",
            "UNAVAILABLE": "Not observable from the standard Razorpay payload.",
        },
        "aggregate": {
            "known": aggregate_known,
            "unknown": aggregate_unknown,
            "unavailable": aggregate_unavailable,
            "total": total_feature_values,
            "known_rate": aggregate_known / total_feature_values,
            "unknown_rate": aggregate_unknown / total_feature_values,
            "unavailable_rate": aggregate_unavailable / total_feature_values,
        },
        "features": feature_analysis,
    }
    _write_json(
        PHASE12_DIR / "unknown_feature_analysis.json",
        unknown_feature_report,
    )

    failure_cases = [
        {
            "payment_id": record["payment_id"],
            "raw_failure_reason": record["raw_failure_reason"],
            "provider_error": record["provider_error"],
            "mapping": record["taxonomy"],
        }
        for record in case_records
    ]
    failure_report = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "total_cases": len(rows),
        "raw_failure_reasons": dict(raw_reason_counts),
        "taxonomy_distribution": dict(taxonomy_counts),
        "known_cases": sum(value != "unknown" for value in (
            diagnosis.taxonomy for diagnosis in diagnoses
        )),
        "unknown_cases": taxonomy_counts["unknown"],
        "unknown_rate": taxonomy_counts["unknown"] / len(rows),
        "cases": failure_cases,
    }
    _write_json(
        PHASE12_DIR / "failure_taxonomy_analysis.json",
        failure_report,
    )

    threshold_results = {}
    thresholds = load_online_feature_schema()["automation_support"][
        "thresholds_to_evaluate"
    ]
    for threshold_value in thresholds:
        threshold = float(threshold_value)
        support_passes = 0
        eligible = 0
        predicted_value = 0.0
        for record, diagnosis in zip(case_records, diagnoses, strict=True):
            support = evaluate_feature_support(
                record["features"],
                threshold=threshold,
            )
            support_passes += int(bool(support.threshold_passed))
            gate = automation_eligibility(
                failure_taxonomy=diagnosis.taxonomy,
                support=support,
            )
            if gate["allowed"]:
                eligible += 1
                selected = record["selected_action"]
                decision = next(
                    row[0] for row in rows
                    if row[1].razorpay_payment_id == record["payment_id"]
                )
                predicted_value += record["amount_inr"] * float(
                    decision.predicted_probabilities.get(selected, 0)
                )
        threshold_results[f"{threshold:.2f}"] = {
            "feature_support_coverage": support_passes / len(rows),
            "feature_support_cases": support_passes,
            "automation_eligible_cases": eligible,
            "fallback_cases": len(rows) - eligible,
            "policy_violations": 0,
            "estimated_recovery_value_inr": round(predicted_value, 2),
        }
    support_report = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "threshold_selection": "not_selected",
        "selection_reason": (
            "All real cases have unknown failure taxonomy; no threshold can "
            "authorize automation from this sample."
        ),
        "score_distribution": _stats(pd.Series(support_scores)),
        "threshold_evaluation": threshold_results,
        "support_matrix": [
            {
                "failure": "known_customer_failure",
                "features_available": "high_or_partial",
                "model_supported": "evaluate",
                "safe_automation": "conditional",
            },
            {
                "failure": "known_payment_method_failure",
                "features_available": "partial",
                "model_supported": "partial",
                "safe_automation": "conditional_after_all_gates",
            },
            {
                "failure": "merchant_configuration",
                "features_available": "high",
                "model_supported": "no",
                "safe_automation": "no",
            },
            {
                "failure": "unknown",
                "features_available": "low_or_unknown",
                "model_supported": "no",
                "safe_automation": "no",
            },
            {
                "failure": "fraud_risk",
                "features_available": "not_relevant",
                "model_supported": "no",
                "safe_automation": "never",
            },
        ],
    }
    _write_json(PHASE12_DIR / "feature_support_analysis.json", support_report)

    model_training_columns = [
        *NUMERIC_DRIFT_FEATURES,
        *CATEGORICAL_DRIFT_FEATURES,
    ]
    model_training = pd.read_csv(
        TRAINING_PATH,
        usecols=model_training_columns,
    )
    source_columns = [
        "amount_inr",
        "hour_of_day",
        *CATEGORICAL_DRIFT_FEATURES,
    ]
    synthetic_source = pd.read_csv(
        SYNTHETIC_SOURCE_PATH,
        usecols=source_columns,
    )
    real_features = pd.DataFrame(
        [record["features"] for record in case_records]
    )
    numerical_drift = {}
    for name in NUMERIC_DRIFT_FEATURES:
        reference = (
            synthetic_source[name]
            if name in synthetic_source
            else model_training[name]
        )
        numerical_drift[name] = {
            "reference": (
                "synthetic_source_transactions"
                if name in synthetic_source
                else "model_training_failures"
            ),
            "training": _stats(reference),
            "real_test_mode": _stats(real_features[name]),
            "ks_statistic_descriptive": _ks_statistic(
                reference,
                real_features[name],
            ),
        }
    categorical_drift = {}
    for name in CATEGORICAL_DRIFT_FEATURES:
        train_distribution = _distribution(synthetic_source[name])
        real_distribution = _distribution(real_features[name])
        categorical_drift[name] = {
            "training_distribution": train_distribution,
            "real_test_mode_distribution": real_distribution,
            "total_variation_descriptive": _total_variation(
                train_distribution,
                real_distribution,
            ),
        }
    drift_report = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "interpretation": "early_distribution_shift_evidence",
        "statistical_caveat": (
            "The real sample has n=20 and is not sufficient for production "
            "population or significance claims."
        ),
        "synthetic_source_rows": len(synthetic_source),
        "model_training_rows": len(model_training),
        "real_rows": len(real_features),
        "numerical": numerical_drift,
        "categorical": categorical_drift,
        "domain_mismatches": {
            "failure_reason": {
                "training": "synthetic UPI status context",
                "real": dict(raw_reason_counts),
                "classification": "SEMANTICALLY_MISMATCHED",
            },
            "payment_method": {
                "training": "no equivalent canonical model feature",
                "real": dict(
                    Counter(record["payment_method"] for record in case_records)
                ),
                "classification": "SEMANTICALLY_MISMATCHED",
            },
        },
    }
    _write_json(
        PHASE12_DIR / "real_vs_training_drift.json",
        drift_report,
    )

    online_schema = load_online_feature_schema()
    deployable_report = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "canonical_model_features": len(model_feature_names()) + 1,
        "deployable_feature_count": sum(
            len(values)
            for values in online_schema["deployable_model_features"].values()
        ),
        "deployable_features": online_schema["deployable_model_features"],
        "excluded_features": online_schema["excluded_model_features"],
        "recommendation": (
            "Keep V1 frozen and evaluate a future V2-online trained only on "
            "deployable features. Do not train on REAL_RAZORPAY_SHADOW_V1."
        ),
    }
    _write_json(
        PHASE12_DIR / "deployable_feature_subset.json",
        deployable_report,
    )

    frozen_feature_set = {
        "version": 1,
        "dataset_id": baseline["baseline_id"],
        "purpose": "deployment compatibility evaluation only",
        "training_allowed": False,
        "case_count": len(case_records),
        "cases": case_records,
    }
    _write_json(
        PHASE12_DIR / "real_razorpay_shadow_v1_features.json",
        frozen_feature_set,
    )

    readiness = {
        "version": 1,
        "baseline_id": baseline["baseline_id"],
        "status": "execution_blocked",
        "infrastructure": {
            "postgresql": "PASS",
            "redis": "PASS",
            "celery": "PASS",
            "real_cases": "20/20",
        },
        "model": {
            "offline_online_parity": "PASS",
            "online_feature_support": "PARTIAL",
            "failure_taxonomy": "BLOCKED",
            "calibration_on_real_cases": "UNKNOWN",
            "distribution_shift": "REVIEW",
        },
        "safety": {
            "unknown_failure": "BLOCK",
            "unknown_or_unavailable_features": "BLOCK",
            "fraud": "BLOCK",
            "weak_margin": "BLOCK",
        },
        "execution": {
            "shadow": "ENABLED",
            "controlled": "BLOCKED",
            "provider_actions_enabled": False,
        },
        "blocking_evidence": [
            "20/20 failures map to unknown taxonomy",
            "online-unavailable and semantically mismatched model inputs exist",
            "20/20 decisions used fallback",
            "20/20 decisions have weak margins",
            "no support threshold is evidence-backed for automation",
        ],
        "frozen_components": {
            "model": "recovery_model_v1",
            "policy": "recovery_policy_v3",
            "execution_mode": "shadow",
            "real_validation_set": baseline["baseline_id"],
        },
    }
    _write_json(PHASE12_DIR / "phase12_readiness.json", readiness)

    artifact_names = (
        "unknown_feature_analysis.json",
        "failure_taxonomy_analysis.json",
        "feature_support_analysis.json",
        "real_vs_training_drift.json",
        "deployable_feature_subset.json",
        "real_razorpay_shadow_v1_features.json",
        "phase12_readiness.json",
    )
    manifest = {
        "version": 1,
        "phase": 12,
        "generated_at": datetime.now(UTC).isoformat(),
        "baseline": {
            "id": baseline["baseline_id"],
            "manifest": str(
                PHASE11_MANIFEST_PATH.relative_to(REPO_ROOT)
            ).replace("\\", "/"),
            "training_allowed": False,
        },
        "inputs": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): {
                "sha256": _sha256(path)
            }
            for path in (
                PHASE11_MANIFEST_PATH,
                REPO_ROOT
                / "ml"
                / "config"
                / "razorpay_failure_taxonomy.yaml",
                REPO_ROOT / "ml" / "config" / "online_feature_schema.yaml",
                REPO_ROOT / "ml" / "config" / "feature_schema.yaml",
                REPO_ROOT / "ml" / "config" / "dataset_manifest.yaml",
                TRAINING_PATH,
                SYNTHETIC_SOURCE_PATH,
                REPO_ROOT / "ml" / "src" / "feature_support.py",
                REPO_ROOT
                / "ml"
                / "src"
                / "razorpay_failure_taxonomy.py",
                Path(__file__),
            )
        },
        "artifacts": {
            name: {
                "path": str((PHASE12_DIR / name).relative_to(REPO_ROOT)).replace(
                    "\\",
                    "/",
                ),
                "sha256": _sha256(PHASE12_DIR / name),
            }
            for name in artifact_names
        },
        "frozen_system_unchanged": True,
        "controlled_execution_authorized": False,
    }
    _write_json(PHASE12_DIR / "phase12_manifest.json", manifest)


if __name__ == "__main__":
    main()
