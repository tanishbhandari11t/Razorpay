from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
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
from app.models.recovery_case import RecoveryCase
from app.models.recovery_job import RecoveryJob
from app.services.shadow_evaluation_gate import evaluate_shadow_gate
from app.services.shadow_monitoring import shadow_metrics
from ml.src.controlled_execution_simulator import (
    simulate_controlled_execution,
)


REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase10"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    initialize_database()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        metrics = shadow_metrics(session)
        gate = evaluate_shadow_gate(session)
        rows = session.execute(
            select(AgentDecision, Payment, RecoveryCase, RecoveryJob)
            .join(Payment, AgentDecision.payment_id == Payment.id)
            .join(
                RecoveryCase,
                AgentDecision.recovery_case_id == RecoveryCase.id,
            )
            .outerjoin(
                RecoveryJob,
                RecoveryJob.recovery_case_id == RecoveryCase.id,
            )
            .where(AgentDecision.execution_mode == "shadow")
            .order_by(AgentDecision.created_at.asc())
        ).all()
    cases = [
        {
            "case_id": decision.recovery_case_id,
            "payment_id": payment.razorpay_payment_id or payment.id,
            "customer_id": payment.customer_id,
            "amount_inr": payment.amount / 100,
            "failure_reason": payment.failure_reason,
            "payment_method": payment.method,
            "payment_timestamp": payment.created_at.isoformat(),
            "features_version": decision.features_version,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "predicted_probabilities": json.dumps(
                decision.predicted_probabilities,
                sort_keys=True,
            ),
            "selected_action": decision.selected_action or "no_action",
            "fallback_used": decision.fallback_used,
            "risk_checks": json.dumps(
                decision.risk_checks,
                sort_keys=True,
            ),
            "risk_checks_passed": decision.risk_checks_passed,
            "decision_margin": decision.decision_margin,
            "failure_class": decision.failure_class,
            "job_status": job.status if job else "missing_job",
            "job_attempts": job.attempts if job else 0,
        }
        for decision, payment, _, job in rows
    ]
    shadow_cases_path = REPORT_DIR / "shadow_cases.csv"
    case_columns = [
        "case_id",
        "payment_id",
        "customer_id",
        "amount_inr",
        "failure_reason",
        "payment_method",
        "payment_timestamp",
        "features_version",
        "model_version",
        "policy_version",
        "predicted_probabilities",
        "selected_action",
        "fallback_used",
        "risk_checks",
        "risk_checks_passed",
        "decision_margin",
        "failure_class",
        "job_status",
        "job_attempts",
    ]
    pd.DataFrame(cases, columns=case_columns).to_csv(
        shadow_cases_path,
        index=False,
    )
    report_path = REPORT_DIR / "phase10_shadow_report.json"
    _write_json(
        report_path,
        {
            "version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "source": "signed_razorpay_webhook_to_durable_worker",
            "metrics": metrics,
            "evaluation_gate": gate,
            "execution_authorized": False,
            "qwen_used": False,
        },
    )
    controlled_path = (
        REPORT_DIR / "controlled_execution_simulation.json"
    )
    simulate_controlled_execution(shadow_cases_path, controlled_path)
    artifacts = [
        report_path,
        shadow_cases_path,
        controlled_path,
        REPORT_DIR / "policy_v4_comparison.json",
        REPORT_DIR / "policy_v4_decisions.csv",
        REPO_ROOT / "ml" / "config" / "execution_gate.yaml",
        REPO_ROOT / "ml" / "config" / "policy_v4.yaml",
        REPO_ROOT / "ml" / "config" / "policy_v4_manifest.yaml",
    ]
    manifest_path = REPORT_DIR / "phase10_report_manifest.json"
    _write_json(
        manifest_path,
        {
            "version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "artifacts": {
                str(path.relative_to(REPO_ROOT)).replace("\\", "/"): _sha256(
                    path
                )
                for path in artifacts
                if path.exists()
            },
        },
    )


if __name__ == "__main__":
    main()
