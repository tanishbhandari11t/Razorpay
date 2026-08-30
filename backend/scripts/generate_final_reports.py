from __future__ import annotations

"""Generate final-phase evaluation and status reports (gates remain locked)."""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sqlalchemy import select


BACKEND_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = BACKEND_ROOT.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.database.connection import get_session, initialize_database
from app.models.agent_decision import AgentDecision
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment
from app.services.outcome_observation import outcome_metrics
from app.services.outcome_state_machine import serialize_outcome
from app.services.recoverai_state import (
    evaluate_recoverai_state,
    progress_to_dict,
)
from ml.src.final_evidence_collector import write_evidence_inventory
from ml.src.generate_final_baseline import write_baseline_manifest
from ml.src.train_final_challenger import write_blocked_model_card
from ml.src.validate_outcomes import certify_outcome_plumbing


REPORT_DIR = REPO_ROOT / "ml" / "reports" / "final"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    initialize_database()
    baseline = write_baseline_manifest()
    validation = certify_outcome_plumbing()
    with get_session() as session:
        metrics = outcome_metrics(session)
        decisions = session.scalars(select(AgentDecision)).all()
        payments = {
            payment.id: payment
            for payment in session.scalars(select(Payment)).all()
        }
        outcomes = session.scalars(select(InterventionOutcome)).all()
    records = []
    for outcome in outcomes:
        serialized = serialize_outcome(outcome)
        decision = next(
            (item for item in decisions if item.id == outcome.agent_decision_id),
            None,
        )
        payment = payments.get(outcome.payment_id)
        snapshot = (decision.features_snapshot if decision else None) or {}
        records.append(
            {
                **serialized,
                "failure_class": decision.failure_class if decision else None,
                "amount_minor": payment.amount if payment else 0,
                "previous_payments": snapshot.get("previous_payments", 0),
            }
        )
    inventory = write_evidence_inventory(
        records,
        shadow_decisions=metrics["shadow_decisions"],
        real_actions_executed=metrics["real_actions_executed"],
    )
    model_card = write_blocked_model_card(evidence_records=records)
    progress = evaluate_recoverai_state(
        evidence_records=records,
        training_eligible_labels=metrics["training_eligible_labels"],
        shadow_decisions=metrics["shadow_decisions"],
        real_failures_observed=metrics["real_failures_observed"],
        execution_mode="shadow",
    )
    state = progress_to_dict(progress)
    evaluation = {
        "version": 1,
        "phase": "final",
        "generated_at": datetime.now(UTC).isoformat(),
        "status": "blocked_awaiting_evidence",
        "execution_mode": "shadow",
        "model_ready": False,
        "phase15_authorized": False,
        "controlled_execution_authorized": False,
        "state": state,
        "evidence_inventory": {
            "real_failures_observed": metrics["real_failures_observed"],
            "shadow_decisions": metrics["shadow_decisions"],
            "real_actions_executed": metrics["real_actions_executed"],
            "observational_recoveries": metrics["observational_recoveries"],
            "attributed_intervention_recoveries": metrics[
                "attributed_intervention_recoveries"
            ],
            "training_eligible_labels": metrics["training_eligible_labels"],
        },
        "baselines": {
            "always_retry": "pending_real_pilot",
            "historical_policy": "pending_real_pilot",
            "v1": "frozen",
            "v2_online": "frozen_not_ready",
            "final_challenger": "blocked",
            "recoverai_final": "blocked",
        },
        "safety": {
            "fraud_actions": 0,
            "duplicate_executions": 0,
            "policy_violations": 0,
            "unsupported_actions": 0,
            "out_of_bound_actions": 0,
        },
        "plumbing_certified": validation["certified"],
        "baseline_freeze": {
            "path": "ml/reports/final/baseline_manifest.json",
            "missing_artifacts": baseline["missing_artifacts"],
        },
        "honest_current_demo": {
            "flow": [
                "real_razorpay_failure",
                "shadow_decision",
                "candidate_audit",
                "safety_gate",
                "block",
                "outcome_observer",
                "audit_trail",
            ],
            "message": (
                "RecoverAI can detect, diagnose, score candidates, and refuse "
                "unsafe execution. It has not earned attributed intervention "
                "labels yet."
            ),
        },
        "model_card_status": model_card["status"],
        "inventory_path": "ml/reports/final/evidence_inventory.json",
    }
    checklist = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "ingestion": {
            "razorpay_webhook": "implemented",
            "hmac_verified": "implemented",
            "event_idempotency": "implemented",
            "payment_persisted": "implemented",
        },
        "ml": {
            "feature_parity": "implemented",
            "challenger_evaluated": "blocked",
            "ope_completed": "phase13_only",
            "model_ready": False,
        },
        "evidence": {
            "real_outcomes_observer": "implemented",
            "observational_ne_intervention": "enforced",
            "phase15_authorized": False,
            "training_eligible_labels": metrics["training_eligible_labels"],
        },
        "qwen": {
            "structured_output": "implemented",
            "financial_authority": False,
            "tools_enabled": False,
            "deterministic_fallback": "implemented",
        },
        "execution": {
            "mode": "shadow",
            "global_kill_switch": "configured",
            "controlled_pilot": "disabled",
            "provider_actions": False,
        },
        "user_actions_required": [
            "Generate more Razorpay Test Mode payment.failed webhooks",
            "Keep Celery outcome observer running through 24h/48h windows",
            "Do not flip phase15_authorized until coverage gate is green",
            "Do not enable controlled execution without model_ready",
        ],
    }
    _write_json(REPORT_DIR / "final_evaluation.json", evaluation)
    _write_json(REPORT_DIR / "recoverai_state.json", state)
    _write_json(REPORT_DIR / "acceptance_checklist.json", checklist)
    _write_json(REPORT_DIR / "outcome_validation.json", validation)
    print(json.dumps({
        "state": state["state"],
        "next_action": state["next_action"],
        "training_eligible_labels": metrics["training_eligible_labels"],
        "inventory": inventory["real_cases"],
        "model_ready": False,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
