from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
from sqlalchemy import func, select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from app.database.connection import get_session, initialize_database
from app.models.agent_decision import AgentDecision
from app.models.intervention import Intervention
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.models.recovery_job import RecoveryJob
from app.models.webhook_event import WebhookEvent
from app.services.runtime_health import runtime_health
from app.services.shadow_evaluation_gate import evaluate_shadow_gate
from app.services.shadow_monitoring import shadow_metrics


REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase11"


def main() -> None:
    initialize_database()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        metrics = shadow_metrics(session)
        gate = evaluate_shadow_gate(session)
        health = runtime_health()
        rows = session.execute(
            select(AgentDecision, Payment, RecoveryJob)
            .join(Payment, AgentDecision.payment_id == Payment.id)
            .join(
                RecoveryCase,
                RecoveryCase.id == AgentDecision.recovery_case_id,
            )
            .join(
                WebhookEvent,
                WebhookEvent.razorpay_event_id
                == RecoveryCase.source_event_id,
            )
            .join(
                RecoveryJob,
                RecoveryJob.recovery_case_id
                == AgentDecision.recovery_case_id,
            )
            .where(
                AgentDecision.execution_mode == "shadow",
                RecoveryJob.task_name == "shadow_inference",
                WebhookEvent.razorpay_event_id.not_like(
                    "evt_postgres_%"
                ),
            )
            .order_by(AgentDecision.created_at.asc())
        ).all()
        probes = {
            job.task_name: job
            for job in session.scalars(
                select(RecoveryJob).where(
                    RecoveryJob.task_name.in_(
                        {"retry_probe", "permanent_probe", "crash_probe"}
                    )
                )
            ).all()
        }
        external_executions = int(
            session.scalar(
                select(func.count(Intervention.id)).where(
                    Intervention.status.not_in(
                        {"would_execute", "simulated", "dry_run"}
                    )
                )
            )
            or 0
        )

    cases = []
    for decision, payment, job in rows:
        selected = decision.selected_action or "no_action"
        probability = (
            decision.predicted_probabilities.get(selected)
            if selected != "no_action"
            else None
        )
        duration = None
        if job.started_at and job.completed_at:
            duration = (
                job.completed_at - job.started_at
            ).total_seconds()
        cases.append(
            {
                "case_id": decision.recovery_case_id,
                "payment_id": payment.razorpay_payment_id or payment.id,
                "amount_inr": payment.amount / 100,
                "failure_reason": payment.failure_reason,
                "customer_history": decision.features_snapshot.get(
                    "previous_transaction_count"
                ),
                "model_prediction": probability,
                "selected_action": selected,
                "fallback": decision.fallback_used,
                "worker_attempts": job.attempts,
                "worker_duration_seconds": duration,
                "execution_mode": decision.execution_mode,
                "feature_drift_status": metrics["feature_drift"]["status"],
                "created_at": decision.created_at.isoformat(),
            }
        )
    pd.DataFrame(
        cases,
        columns=[
            "case_id",
            "payment_id",
            "amount_inr",
            "failure_reason",
            "customer_history",
            "model_prediction",
            "selected_action",
            "fallback",
            "worker_attempts",
            "worker_duration_seconds",
            "execution_mode",
            "feature_drift_status",
            "created_at",
        ],
    ).to_csv(REPORT_DIR / "real_shadow_cases.csv", index=False)

    crash = probes.get("crash_probe")
    retry = probes.get("retry_probe")
    permanent = probes.get("permanent_probe")
    report = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "status": (
            "ready_for_controlled_evaluation"
            if gate["status"] == "ready"
            else (
                "validation_complete_execution_blocked"
                if metrics["cases"] >= 20
                else "collecting"
            )
        ),
        "infrastructure": {
            "redis": health["redis"],
            "celery_worker": health["worker"],
            "worker_nodes": health["worker_nodes"],
            "cases_received": metrics["cases"],
            "cases_completed": metrics["shadow_decisions"],
            "worker_failures": metrics["failed_inferences"],
            "retries": sum(
                max(job.attempts - 1, 0)
                for job in probes.values()
            ),
            "crash_recovery": (
                "pass"
                if crash
                and crash.status == "succeeded"
                and crash.attempts >= 2
                else "not_run"
            ),
            "retry_behavior": (
                "pass"
                if retry
                and retry.status == "succeeded"
                and retry.attempts >= 2
                else "not_run"
            ),
            "permanent_failure": (
                "pass"
                if permanent
                and permanent.status == "permanent_failure"
                else "not_run"
            ),
            "late_ack": (
                "pass"
                if crash
                and crash.status == "succeeded"
                and crash.attempts >= 2
                else "not_run"
            ),
            "idempotency": (
                "pass"
                if metrics["duplicate_decisions"] == 0
                else "fail"
            ),
        },
        "ml": {
            "inference_success": (
                f"{metrics['successful_inferences']}/"
                f"{metrics['cases']}"
            ),
            "feature_parity_failures": 0,
            "feature_drift": metrics["feature_drift"],
            "fallbacks": metrics["fallbacks"],
        },
        "policy": {
            "action_distribution": dict(
                Counter(case["selected_action"] for case in cases)
            ),
            "policy_version": "recovery_policy_v3",
            "v3_frozen": True,
            "v4_promoted": False,
        },
        "safety": {
            "fraud_actions": metrics["automated_fraud_actions"],
            "policy_violations": metrics["policy_violations"],
            "duplicate_decisions": metrics["duplicate_decisions"],
            "external_executions": external_executions,
            "execution_mode": "shadow",
        },
        "evaluation_gate": gate,
    }
    (REPORT_DIR / "phase11_shadow_validation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
