from __future__ import annotations

import hashlib
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.ml.model_loader import ModelBundle, load_model_bundle
from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.intervention import Intervention
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.candidate_audit import build_candidate_audit
from app.services.execution_gate import check_execution_gate
from app.services.failure_mapping import map_razorpay_failure_reason
from app.services.features.builder import build_online_features
from app.services.recovery_agent import extract_agent_plan, run_agent_for_decision


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.failure_classifier import load_intervention_policy
from ml.src.feature_support import (
    automation_eligibility,
    evaluate_feature_support,
)
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    decide_recovery_action,
    load_action_matrix,
)
from ml.src.razorpay_failure_taxonomy import classify_razorpay_failure
from ml.src.policies.stopping_rules import (
    InterventionAttempt,
    RecoveryPolicyContext,
)
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    load_support_policy_config,
)


POLICY_V3_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "policy_v3_manifest.yaml"
)
CONTEXT_SUPPORT_PATH = REPO_ROOT / "ml" / "reports" / "context_support.csv"
FEATURES_VERSION = "temporal_recovery_features_v1"
EXECUTION_MODE = "shadow"


class ShadowInferenceError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@lru_cache(maxsize=1)
def _support_index() -> SupportIndex:
    return SupportIndex(
        pd.read_csv(CONTEXT_SUPPORT_PATH),
        load_support_policy_config(),
    )


def _predict_candidates(
    features: dict[str, Any],
    bundle: ModelBundle,
) -> dict[str, float]:
    actions = list(load_intervention_policy()["interventions"])
    rows = []
    for action in actions:
        row = dict(features)
        row["chosen_intervention"] = action
        rows.append({name: row[name] for name in bundle.raw_feature_names})
    frame = pd.DataFrame(rows, columns=bundle.raw_feature_names)
    transformed = bundle.preprocessor.transform(frame)
    raw = bundle.model.predict_proba(transformed)[:, 1]
    calibrated = np.asarray(bundle.calibrator.predict(raw), dtype=float)
    if (
        calibrated.shape != (len(actions),)
        or not np.isfinite(calibrated).all()
        or not ((calibrated >= 0) & (calibrated <= 1)).all()
    ):
        raise ShadowInferenceError("Model produced invalid probabilities")
    return {
        action: float(probability)
        for action, probability in zip(actions, calibrated, strict=True)
    }


def _policy_context(
    session: Session,
    recovery_case: RecoveryCase,
    payment: Payment,
    customer: Customer,
    features: dict[str, Any],
) -> RecoveryPolicyContext:
    attempts = session.scalars(
        select(Intervention)
        .where(Intervention.payment_id == payment.id)
        .order_by(Intervention.created_at.asc())
    ).all()
    return RecoveryPolicyContext(
        case_id=recovery_case.id,
        payment_id=payment.id,
        amount_inr=float(payment.amount) / 100,
        payment_status=payment.status,
        failure_reason=map_razorpay_failure_reason(
            payment.failure_reason,
            fraud_flag=int(features["fraud_flag"]),
        ),
        fraud_flag=int(features["fraud_flag"]),
        case_created_at=recovery_case.created_at,
        now=datetime.now(UTC),
        customer_contact_available=bool(customer.phone or customer.email),
        customer_opted_out=False,
        valid_payment_context=bool(payment.razorpay_payment_id and payment.amount > 0),
        attempts=[
            InterventionAttempt(
                action=attempt.type,
                status=attempt.status,
                executed_at=attempt.created_at,
                cost_inr=float(attempt.cost),
            )
            for attempt in attempts
        ],
    )


def _support(
    features: dict[str, Any],
    probabilities: dict[str, float],
) -> dict[str, CandidateSupport]:
    if int(features["has_prior_history"]) == 0:
        return {
            action: CandidateSupport(supported=False)
            for action in probabilities
        }
    index = _support_index()
    support: dict[str, CandidateSupport] = {}
    for action in probabilities:
        evidence = index.evidence(features, action)
        support[action] = CandidateSupport(
            supported=evidence.supported,
            action_count=evidence.action_count,
            effective_sample_size=evidence.effective_sample_size,
        )
    return support


def _decision_margin(probabilities: dict[str, float]) -> float | None:
    values = sorted(probabilities.values(), reverse=True)
    return values[0] - values[1] if len(values) > 1 else None


def _json_features(features: dict[str, Any]) -> dict[str, Any]:
    return {
        name: value.isoformat() if isinstance(value, datetime) else value
        for name, value in features.items()
    }


def evaluate_shadow_case(
    session: Session,
    recovery_case_id: str,
) -> tuple[AgentDecision, bool]:
    bundle = load_model_bundle()
    decision_key = (
        f"{recovery_case_id}:{bundle.policy_version}:{EXECUTION_MODE}"
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
    if payment.status.lower() in {"captured", "recovered", "paid"}:
        raise ShadowInferenceError("Payment is already recovered")
    if payment.status.lower() != "failed":
        raise ShadowInferenceError("Only failed payments can be evaluated")

    features = build_online_features(session, recovery_case_id)
    probabilities = _predict_candidates(features, bundle)
    context = _policy_context(
        session,
        recovery_case,
        payment,
        customer,
        features,
    )
    taxonomy = classify_razorpay_failure(
        payment.failure_reason,
        fraud_flag=int(features["fraud_flag"]),
    )
    feature_support = evaluate_feature_support(features)
    compatibility_gate = automation_eligibility(
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
    compatibility_fallback = not compatibility_gate["allowed"]
    effective_action = (
        "escalate_to_merchant"
        if compatibility_fallback
        else decision.selected_action
    )
    effective_decision_type = (
        "fallback"
        if compatibility_fallback
        else decision.decision_type.value
    )
    execution_gate = check_execution_gate(effective_action)
    if execution_gate.allowed:
        raise ShadowInferenceError(
            "Shadow inference refuses an execution-enabled configuration"
        )
    no_history_fallback = int(features["has_prior_history"]) == 0
    decision_reasons = list(decision.reasons)
    if compatibility_fallback:
        decision_reasons.extend(
            [
                "phase12_compatibility_fallback",
                *compatibility_gate["reasons"],
            ]
        )
    if no_history_fallback:
        decision_reasons.append("no_prior_history_shadow_fallback")
    record = AgentDecision(
        recovery_case_id=recovery_case.id,
        payment_id=payment.id,
        decision_key=decision_key,
        model_version=bundle.model_version,
        policy_version=bundle.policy_version,
        features_version=FEATURES_VERSION,
        policy_manifest_sha256=_sha256(POLICY_V3_MANIFEST_PATH),
        execution_mode=EXECUTION_MODE,
        inference_status="completed",
        decision_type=effective_decision_type,
        selected_action=effective_action,
        candidate_actions={
            action: asdict(candidate)
            for action, candidate in decision.candidates.items()
        },
        predicted_probabilities=decision.probabilities,
        expected_values=decision.expected_values,
        features_snapshot=_json_features(features),
        decision_reasons=decision_reasons,
        decision_margin=_decision_margin(decision.probabilities),
        failure_class=decision.failure_class,
        fallback_used=(
            decision.fallback_used
            or no_history_fallback
            or compatibility_fallback
        ),
        risk_checks=[
            {
                "name": "policy_v3",
                "passed": decision.risk_checks_passed,
            },
            {
                "name": "strict_temporal_cutoff",
                "passed": True,
            },
            {
                "name": "payment_still_failed",
                "passed": payment.status.lower() == "failed",
            },
            {
                "name": "execution_gate_blocked",
                "passed": not execution_gate.allowed,
                "reason": execution_gate.reason,
            },
            {
                "name": "phase12_compatibility_gate",
                "passed": compatibility_gate["allowed"],
                "reasons": compatibility_gate["reasons"],
                "taxonomy_id": "razorpay_failure_taxonomy_v1",
                "raw_failure_reason": taxonomy.raw_reason,
                "normalized_failure_reason": taxonomy.normalized_reason,
                "failure_taxonomy": taxonomy.taxonomy,
                "failure_subtype": taxonomy.subtype,
                "failure_taxonomy_state": taxonomy.state,
                "mapping_confidence": taxonomy.confidence,
                "matched_rule": taxonomy.matched_rule,
                "failure_retryable": taxonomy.retryable,
                "safe_automation": taxonomy.safe_automation,
                "taxonomy_execution_allowed": taxonomy.execution_allowed,
                "feature_support_score": feature_support.score,
                "feature_support_threshold": feature_support.threshold,
                "known_feature_count": feature_support.known,
                "unknown_feature_count": feature_support.unknown,
                "unavailable_feature_count": feature_support.unavailable,
                "feature_support_denominator": feature_support.total,
            },
        ],
        risk_checks_passed=(
            decision.risk_checks_passed
            and compatibility_gate["allowed"]
        ),
        dry_run=True,
    )
    # Shadow agent: draft + validate communication. Never executes.
    run_agent_for_decision(session, record)
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


def serialize_shadow_decision(
    record: AgentDecision,
    *,
    idempotent_replay: bool,
) -> dict[str, Any]:
    audit = build_candidate_audit(
        payment_id=record.payment_id,
        predicted_probabilities=record.predicted_probabilities or {},
        expected_values=record.expected_values or {},
        winner=record.selected_action,
        executed=False,
        blocked_reason="execution_mode_shadow",
    )
    agent_plan = extract_agent_plan(record)
    return {
        "id": record.id,
        "recovery_case_id": record.recovery_case_id,
        "model_version": record.model_version,
        "policy_version": record.policy_version,
        "features_version": record.features_version,
        "execution_mode": record.execution_mode,
        "predictions": record.predicted_probabilities,
        "candidate_audit": audit,
        "winner": audit["winner"],
        "executed": False,
        "blocked_reason": audit["blocked_reason"],
        "candidate_actions": record.candidate_actions,
        "selected_action": record.selected_action,
        "decision_type": record.decision_type,
        "decision_reason": record.decision_reasons,
        "decision_margin": record.decision_margin,
        "failure_class": record.failure_class,
        "fallback_used": record.fallback_used,
        "risk_checks": record.risk_checks,
        "agent_plan": agent_plan,
        "would_execute": record.selected_action,
        "idempotent_replay": idempotent_replay,
        "created_at": record.created_at.isoformat(),
    }
