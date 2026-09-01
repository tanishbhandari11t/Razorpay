from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.models.recovery_job import RecoveryJob
from app.models.webhook_event import WebhookEvent


REPO_ROOT = Path(__file__).resolve().parents[3]
TRAINING_DATA_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "logging_policy_dataset.csv"
)
DRIFT_CATEGORICALS = (
    "transaction_type",
    "merchant_category",
    "device_type",
    "network_type",
    "sender_state",
    "sender_bank",
)


@lru_cache(maxsize=1)
def _training_reference() -> dict[str, Any]:
    empty = {
        "amount_mean": 0.0,
        "no_history_rate": 0.0,
        "hour_distribution": {hour: 0.0 for hour in range(24)},
        "categories": {name: set() for name in DRIFT_CATEGORICALS},
        "available": False,
    }
    if not TRAINING_DATA_PATH.is_file():
        return empty
    columns = [
        "amount_inr",
        "has_prior_history",
        "hour_of_day",
        *DRIFT_CATEGORICALS,
    ]
    training = pd.read_csv(TRAINING_DATA_PATH, usecols=columns)
    return {
        "amount_mean": float(training["amount_inr"].mean()),
        "no_history_rate": float(
            1 - training["has_prior_history"].astype(float).mean()
        ),
        "hour_distribution": (
            training["hour_of_day"]
            .astype(int)
            .value_counts(normalize=True)
            .reindex(range(24), fill_value=0.0)
            .to_dict()
        ),
        "categories": {
            name: set(training[name].astype(str))
            for name in DRIFT_CATEGORICALS
        },
        "available": True,
    }


def shadow_metrics(session: Session) -> dict[str, Any]:
    all_jobs = session.scalars(
        select(RecoveryJob)
        .join(
            RecoveryCase,
            RecoveryJob.recovery_case_id == RecoveryCase.id,
        )
        .join(
            WebhookEvent,
            RecoveryCase.source_event_id
            == WebhookEvent.razorpay_event_id,
        )
        .where(
            RecoveryJob.execution_mode == "shadow",
            RecoveryJob.task_name == "shadow_inference",
            RecoveryJob.policy_version == "recovery_policy_v3",
            WebhookEvent.razorpay_event_id.not_like("evt_postgres_%"),
        )
    ).all()
    all_job_statuses = Counter(job.status for job in all_jobs)
    all_job_cases = {job.recovery_case_id for job in all_jobs}
    rows = session.execute(
        select(AgentDecision, Payment, RecoveryCase, WebhookEvent)
        .join(Payment, AgentDecision.payment_id == Payment.id)
        .join(
            RecoveryCase,
            AgentDecision.recovery_case_id == RecoveryCase.id,
        )
        .join(
            WebhookEvent,
            RecoveryCase.source_event_id == WebhookEvent.razorpay_event_id,
        )
        .where(
            AgentDecision.execution_mode == "shadow",
            AgentDecision.policy_version == "recovery_policy_v3",
            WebhookEvent.razorpay_event_id.not_like("evt_postgres_%"),
        )
        .order_by(AgentDecision.created_at.desc())
    ).all()
    if not rows:
        return {
            "execution_mode": "shadow",
            "cases": len(all_job_cases),
            "at_risk_revenue": 0.0,
            "shadow_decisions": 0,
            "successful_inferences": all_job_statuses["succeeded"],
            "failed_inferences": (
                all_job_statuses["failed"]
                + all_job_statuses["permanent_failure"]
            ),
            "job_statuses": dict(all_job_statuses),
            "fallbacks": 0,
            "blocked_actions": 0,
            "predicted_recovery_value": 0.0,
            "observed_recoveries": 0,
            "observed_recovery_rate": None,
            "mean_selected_probability": None,
            "mean_decision_margin": None,
            "weak_margin_count": 0,
            "action_distribution": {},
            "failure_taxonomy": {},
            "unknown_failure_rate": None,
            "feature_missing_rate": None,
            "policy_violations": 0,
            "automated_fraud_actions": 0,
            "duplicate_decisions": 0,
            "duplicate_webhook_deliveries": 0,
            "feature_drift": {
                "status": "collecting",
                "reasons": ["no_shadow_cases"],
            },
        }

    decisions = [decision for decision, _, _, _ in rows]
    payments = [payment for _, payment, _, _ in rows]
    events = [event for _, _, _, event in rows]
    selected_probabilities = [
        float(decision.predicted_probabilities[decision.selected_action])
        for decision in decisions
        if decision.selected_action
        and decision.selected_action in decision.predicted_probabilities
    ]
    margins = [
        float(decision.decision_margin)
        for decision in decisions
        if decision.decision_margin is not None
    ]
    unique_payments = {payment.id: payment for payment in payments}
    predicted_value = sum(
        float(decision.features_snapshot.get("amount_inr", 0))
        * float(decision.predicted_probabilities.get(decision.selected_action, 0))
        for decision in decisions
        if decision.selected_action
    )
    observed_recoveries = sum(
        payment.status.lower() in {"captured", "recovered", "paid"}
        for payment in payments
    )

    reference = _training_reference()
    snapshots = [decision.features_snapshot for decision in decisions]
    amounts = [float(snapshot["amount_inr"]) for snapshot in snapshots]
    no_history_rate = sum(
        int(snapshot["has_prior_history"]) == 0 for snapshot in snapshots
    ) / len(snapshots)
    unknown_values = sum(
        str(snapshot.get(name, "UNKNOWN")) == "UNKNOWN"
        for snapshot in snapshots
        for name in DRIFT_CATEGORICALS
    )
    unknown_rate = unknown_values / (len(snapshots) * len(DRIFT_CATEGORICALS))
    unseen_values = sum(
        str(snapshot.get(name, "UNKNOWN")) not in reference["categories"][name]
        for snapshot in snapshots
        for name in DRIFT_CATEGORICALS
    )
    unseen_rate = unseen_values / (len(snapshots) * len(DRIFT_CATEGORICALS))
    missing_values = sum(
        value is None
        for snapshot in snapshots
        for value in snapshot.values()
    )
    feature_values = sum(len(snapshot) for snapshot in snapshots)
    feature_missing_rate = (
        missing_values / feature_values if feature_values else 1.0
    )
    unknown_failures = sum(
        (decision.failure_class or "unknown") == "unknown"
        for decision in decisions
    )
    policy_violations = sum(
        bool(decision.selected_action) and not decision.risk_checks_passed
        for decision in decisions
    )
    automated_fraud_actions = sum(
        decision.failure_class == "fraud_risk"
        and decision.selected_action
        not in {None, "escalate_to_merchant", "no_action"}
        for decision in decisions
    )
    decision_identities = {
        (
            decision.recovery_case_id,
            decision.model_version,
            decision.policy_version,
            decision.execution_mode,
        )
        for decision in decisions
    }
    amount_ratio = (
        (sum(amounts) / len(amounts)) / reference["amount_mean"]
        if reference["amount_mean"]
        else 0.0
    )
    shadow_hour_distribution = Counter(
        int(snapshot["hour_of_day"]) for snapshot in snapshots
    )
    hour_total_variation = 0.5 * sum(
        abs(
            shadow_hour_distribution.get(hour, 0) / len(snapshots)
            - reference["hour_distribution"][hour]
        )
        for hour in range(24)
    )
    reasons = []
    if not reference.get("available", True):
        reasons.append("training_reference_missing")
    if unknown_rate > 0.2:
        reasons.append("high_unknown_category_rate")
    if unseen_rate > 0.2:
        reasons.append("high_unseen_category_rate")
    if amount_ratio < 0.5 or amount_ratio > 2:
        reasons.append("amount_distribution_shift")
    if abs(no_history_rate - reference["no_history_rate"]) > 0.2:
        reasons.append("customer_history_shift")
    if hour_total_variation > 0.35:
        reasons.append("hour_distribution_shift")
    drift_status = "review" if reasons else "aligned"
    if len(snapshots) < 20:
        drift_status = "collecting"
        reasons.append("minimum_20_cases_required")
    elif unknown_rate > 0.5 or unseen_rate > 0.5:
        drift_status = "stop"

    return {
        "execution_mode": "shadow",
        "cases": len(all_job_cases),
        "at_risk_revenue": round(
            sum(payment.amount for payment in unique_payments.values()) / 100,
            2,
        ),
        "shadow_decisions": len(decisions),
        "successful_inferences": all_job_statuses["succeeded"],
        "failed_inferences": (
            all_job_statuses["failed"]
            + all_job_statuses["permanent_failure"]
        ),
        "job_statuses": dict(all_job_statuses),
        "fallbacks": sum(decision.fallback_used for decision in decisions),
        "blocked_actions": sum(
            decision.decision_type in {"block", "stop"}
            for decision in decisions
        ),
        "predicted_recovery_value": round(predicted_value, 2),
        "observed_recoveries": observed_recoveries,
        "observed_recovery_rate": observed_recoveries / len(decisions),
        "mean_selected_probability": (
            sum(selected_probabilities) / len(selected_probabilities)
            if selected_probabilities
            else None
        ),
        "mean_decision_margin": (
            sum(margins) / len(margins) if margins else None
        ),
        "weak_margin_count": sum(margin < 0.05 for margin in margins),
        "action_distribution": dict(
            Counter(
                decision.selected_action or "no_action"
                for decision in decisions
            )
        ),
        "failure_taxonomy": dict(
            Counter(decision.failure_class or "unknown" for decision in decisions)
        ),
        "unknown_failure_rate": unknown_failures / len(decisions),
        "feature_missing_rate": feature_missing_rate,
        "policy_violations": policy_violations,
        "automated_fraud_actions": automated_fraud_actions,
        "duplicate_decisions": len(decisions) - len(decision_identities),
        "duplicate_webhook_deliveries": sum(
            max(int(event.delivery_count) - 1, 0)
            for event in {event.id: event for event in events}.values()
        ),
        "razorpay_failure_reasons": dict(
            Counter(payment.failure_reason or "unknown" for payment in payments)
        ),
        "payment_method_distribution": dict(
            Counter(payment.method or "unknown" for payment in payments)
        ),
        "hour_distribution": {
            str(hour): shadow_hour_distribution.get(hour, 0)
            for hour in range(24)
        },
        "feature_drift": {
            "status": drift_status,
            "reasons": reasons,
            "unknown_category_rate": round(unknown_rate, 6),
            "unseen_category_rate": round(unseen_rate, 6),
            "shadow_amount_mean": round(sum(amounts) / len(amounts), 2),
            "training_amount_mean": round(reference["amount_mean"], 2),
            "amount_mean_ratio": round(amount_ratio, 6),
            "hour_total_variation": round(hour_total_variation, 6),
            "shadow_no_history_rate": round(no_history_rate, 6),
            "training_no_history_rate": round(
                reference["no_history_rate"],
                6,
            ),
        },
    }
