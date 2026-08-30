from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ml.model_loader_v2_online import load_v2_online_model_bundle
from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.execution_gate import check_execution_gate, load_execution_gate
from app.services.features.builder import build_online_features
from app.services.recovery_inference import (
    ShadowInferenceError,
    _json_features,
    _policy_context,
    _predict_candidates,
    _sha256,
    _support,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
SHADOW_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "phase13_shadow.yaml"
MODEL_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "recovery_model_v2_online_manifest.yaml"
)


def load_v2_shadow_config() -> dict[str, Any]:
    config = yaml.safe_load(SHADOW_CONFIG_PATH.read_text(encoding="utf-8"))
    if config["identity"]["execution_mode"] != "shadow":
        raise ShadowInferenceError("V2-online challenger must remain shadow")
    if config["safety"]["provider_actions_enabled"]:
        raise ShadowInferenceError("V2-online provider actions must be disabled")
    if config["safety"]["controlled_execution_authorized"]:
        raise ShadowInferenceError(
            "V2-online controlled execution must remain unauthorized"
        )
    return config


def evaluate_v2_online_shadow_case(
    session: Session,
    recovery_case_id: str,
) -> tuple[AgentDecision, bool]:
    shadow_config = load_v2_shadow_config()
    global_gate = load_execution_gate()
    if (
        shadow_config["safety"]["refuse_if_global_execution_not_shadow"]
        and global_gate["execution"]["mode"] != "shadow"
    ):
        raise ShadowInferenceError(
            "V2-online challenger refuses non-shadow global execution"
        )
    bundle = load_v2_online_model_bundle()
    execution_mode = str(shadow_config["identity"]["execution_mode"])
    decision_key = (
        f"{recovery_case_id}:{bundle.policy_version}:{execution_mode}"
    )
    existing = session.scalar(
        select(AgentDecision).where(
            AgentDecision.decision_key == decision_key
        )
    )
    if existing is not None:
        return existing, True
    row = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .where(RecoveryCase.id == recovery_case_id)
    ).one_or_none()
    if row is None:
        raise ShadowInferenceError("Recovery case not found")
    recovery_case, payment, customer = row
    if payment.status.lower() != "failed":
        raise ShadowInferenceError(
            "V2-online shadow evaluates failed payments only"
        )

    features = build_online_features(session, recovery_case_id)
    probabilities = _predict_candidates(features, bundle)
    context = _policy_context(
        session,
        recovery_case,
        payment,
        customer,
        features,
    )

    from ml.src.failure_classifier import load_intervention_policy
    from ml.src.feature_support import (
        automation_eligibility,
        evaluate_feature_support,
    )
    from ml.src.policies.recovery_policy import (
        decide_recovery_action,
        load_action_matrix,
    )
    from ml.src.policies.decision_margin import apply_decision_margin
    from ml.src.razorpay_failure_taxonomy import (
        classify_razorpay_failure,
    )

    taxonomy = classify_razorpay_failure(
        payment.failure_reason,
        fraud_flag=int(features["fraud_flag"]),
    )
    feature_support = evaluate_feature_support(features)
    compatibility = automation_eligibility(
        failure_taxonomy=taxonomy.taxonomy,
        support=feature_support,
    )
    decision = decide_recovery_action(
        context,
        probabilities,
        _support(features, probabilities),
        policy=load_intervention_policy(),
        action_matrix=load_action_matrix(),
    )
    margin_gate = apply_decision_margin(
        decision,
        fallback_action=str(
            shadow_config["decision_margin"]["fallback_action"]
        ),
        threshold_inr=float(
            shadow_config["decision_margin"][
                "minimum_expected_value_margin_inr"
            ]
        ),
    )
    compatibility_fallback = not compatibility["allowed"]
    selected_action = (
        "escalate_to_merchant"
        if compatibility_fallback
        else margin_gate.selected_action
    )
    decision_type = (
        "fallback"
        if compatibility_fallback
        else (
            "fallback"
            if margin_gate.fallback_triggered
            else decision.decision_type.value
        )
    )
    execution_gate = check_execution_gate(selected_action)
    if execution_gate.allowed:
        raise ShadowInferenceError(
            "V2-online shadow refuses an execution-enabled configuration"
        )
    reasons = list(decision.reasons)
    reasons.append(margin_gate.reason)
    if compatibility_fallback:
        reasons.extend(
            ["phase12_compatibility_fallback", *compatibility["reasons"]]
        )
    no_history_fallback = int(features["has_prior_history"]) == 0
    if no_history_fallback:
        reasons.append("no_prior_history_shadow_fallback")
    record = AgentDecision(
        recovery_case_id=recovery_case.id,
        payment_id=payment.id,
        decision_key=decision_key,
        model_version=bundle.model_version,
        policy_version=bundle.policy_version,
        features_version=str(
            shadow_config["identity"]["feature_version"]
        ),
        policy_manifest_sha256=_sha256(MODEL_MANIFEST_PATH),
        execution_mode=execution_mode,
        inference_status="completed",
        decision_type=decision_type,
        selected_action=selected_action,
        candidate_actions={
            action: asdict(candidate)
            for action, candidate in decision.candidates.items()
        },
        predicted_probabilities=decision.probabilities,
        expected_values=decision.expected_values,
        features_snapshot=_json_features(features),
        decision_reasons=reasons,
        decision_margin=margin_gate.decision_margin_inr,
        failure_class=decision.failure_class,
        fallback_used=(
            decision.fallback_used
            or compatibility_fallback
            or no_history_fallback
            or margin_gate.fallback_triggered
        ),
        risk_checks=[
            {
                "name": "policy_v3_unchanged_v2_online_probabilities",
                "passed": decision.risk_checks_passed,
            },
            {
                "name": "phase12_compatibility_gate",
                "passed": compatibility["allowed"],
                "reasons": compatibility["reasons"],
                "taxonomy_id": "razorpay_failure_taxonomy_v1",
                "raw_failure_reason": taxonomy.raw_reason,
                "normalized_failure_reason": taxonomy.normalized_reason,
                "failure_taxonomy": taxonomy.taxonomy,
                "failure_subtype": taxonomy.subtype,
                "failure_taxonomy_state": taxonomy.state,
                "matched_rule": taxonomy.matched_rule,
                "feature_support_score": feature_support.score,
                "known_feature_count": feature_support.known,
                "unknown_feature_count": feature_support.unknown,
                "unavailable_feature_count": feature_support.unavailable,
                "feature_support_denominator": feature_support.total,
            },
            {
                "name": "phase13_decision_margin",
                "passed": True,
                "margin_passed": not margin_gate.fallback_triggered,
                "fallback_action": shadow_config["decision_margin"][
                    "fallback_action"
                ],
                "best_action_value_inr": (
                    margin_gate.best_action_value_inr
                ),
                "fallback_value_inr": margin_gate.fallback_value_inr,
                "decision_margin_inr": margin_gate.decision_margin_inr,
                "minimum_margin_inr": margin_gate.threshold_inr,
                "reason": margin_gate.reason,
            },
            {
                "name": "execution_gate_blocked",
                "passed": not execution_gate.allowed,
                "reason": execution_gate.reason,
            },
            {
                "name": "no_intervention_write",
                "passed": True,
            },
        ],
        risk_checks_passed=(
            decision.risk_checks_passed and compatibility["allowed"]
        ),
        dry_run=True,
    )
    session.add(record)
    try:
        session.flush()
    except IntegrityError:
        session.rollback()
        existing = session.scalar(
            select(AgentDecision).where(
                AgentDecision.decision_key == decision_key
            )
        )
        if existing is None:
            raise
        return existing, True
    return record, False


def serialize_v2_online_shadow_decision(
    record: AgentDecision,
    *,
    idempotent_replay: bool,
) -> dict[str, Any]:
    return {
        "id": record.id,
        "recovery_case_id": record.recovery_case_id,
        "model_version": record.model_version,
        "policy_version": record.policy_version,
        "features_version": record.features_version,
        "execution_mode": record.execution_mode,
        "predictions": record.predicted_probabilities,
        "selected_action": record.selected_action,
        "decision_type": record.decision_type,
        "decision_reasons": record.decision_reasons,
        "decision_margin": record.decision_margin,
        "failure_class": record.failure_class,
        "fallback_used": record.fallback_used,
        "risk_checks": record.risk_checks,
        "executed": False,
        "idempotent_replay": idempotent_replay,
        "created_at": record.created_at.isoformat(),
    }
