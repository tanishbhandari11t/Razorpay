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
from app.models.intervention_outcome import InterventionOutcome
from app.models.payment import Payment
from app.services.outcome_observation import outcome_metrics
from app.services.outcome_state_machine import (
    load_real_outcome_schema,
    serialize_outcome,
)
from ml.src.build_real_outcome_dataset import write_real_observed_outcomes
from ml.src.evidence_coverage import write_evidence_coverage_report
from ml.src.record_recovery_outcome import retraining_gate
from ml.src.train_recovery_model_v3_challenger import write_phase15_readiness
from ml.src.validate_outcomes import (
    certify_outcome_plumbing,
    evaluate_evidence_coverage,
)


REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase14"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    initialize_database()
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with get_session() as session:
        decisions = session.scalars(select(AgentDecision)).all()
        payments = session.scalars(select(Payment)).all()
        outcomes = session.scalars(
            select(InterventionOutcome).order_by(
                InterventionOutcome.created_at
            )
        ).all()
        metrics = outcome_metrics(session)
    tracked_decisions = {outcome.agent_decision_id for outcome in outcomes}
    schema = load_real_outcome_schema()
    training_config = schema["training_eligibility"]
    eligible = [
        outcome
        for outcome in outcomes
        if outcome.attempted
        and outcome.data_source == training_config["require_data_source"]
        and outcome.outcome_state in {"recovered", "no_recovery_observed"}
        and not (
            outcome.natural_recovery_observed
            and outcome.outcome_state != "recovered"
        )
    ]
    action_counts = {
        action: sum(outcome.action == action for outcome in eligible)
        for action in (
            "retry_payment",
            "payment_link",
            "whatsapp_reminder",
            "escalate_to_merchant",
        )
    }
    class_counts = {
        state: sum(outcome.outcome_state == state for outcome in eligible)
        for state in ("recovered", "no_recovery_observed")
    }
    coverage_passed = (
        all(class_counts.values())
        and all(
            count
            >= int(training_config["coverage_requirements"][
                "minimum_rows_per_action"
            ])
            for count in action_counts.values()
        )
    )
    evidence_sufficient = (
        len(eligible) >= int(training_config["minimum_rows_for_retraining"])
        and coverage_passed
    )
    columns = [
        "id",
        "decision_id",
        "intervention_id",
        "payment_id",
        "recovery_case_id",
        "action",
        "decision_probability",
        "decision_margin",
        "model_version",
        "policy_version",
        "execution_mode",
        "attempted",
        "attempted_at",
        "failure_timestamp",
        "payment_status_after_24h",
        "payment_status_after_48h",
        "outcome_state",
        "outcome_at",
        "payment_recovered",
        "recovered_amount_minor",
        "recovery_timestamp",
        "time_to_recovery_seconds",
        "observation_window_starts_at",
        "observation_window_ends_at",
        "outcome_source",
        "data_source",
        "natural_recovery_observed",
        "label_kind",
    ]
    frame = pd.DataFrame(
        [serialize_outcome(outcome) for outcome in outcomes],
        columns=columns,
    )
    cases_path = REPORT_DIR / "real_outcome_cases.csv"
    frame.to_csv(cases_path, index=False)
    dataset_split = write_real_observed_outcomes(frame)
    gate = retraining_gate(
        metrics["training_eligible_labels"],
        coverage_passed=coverage_passed,
    )
    validation = certify_outcome_plumbing()
    validation_path = REPORT_DIR / "outcome_validation.json"
    _write_json(validation_path, validation)
    payment_by_id = {payment.id: payment for payment in payments}
    decision_by_id = {decision.id: decision for decision in decisions}
    coverage_records = []
    for outcome in outcomes:
        serialized = serialize_outcome(outcome)
        decision = decision_by_id.get(outcome.agent_decision_id)
        payment = payment_by_id.get(outcome.payment_id)
        snapshot = (decision.features_snapshot if decision else None) or {}
        coverage_records.append(
            {
                **serialized,
                "failure_class": (
                    decision.failure_class if decision else None
                ),
                "amount_minor": payment.amount if payment else 0,
                "previous_payments": snapshot.get("previous_payments", 0),
            }
        )
    coverage = evaluate_evidence_coverage(coverage_records)
    coverage_path = REPORT_DIR / "outcome_coverage.json"
    evidence_report = write_evidence_coverage_report(coverage_records)
    _write_json(coverage_path, coverage)
    phase15 = write_phase15_readiness(coverage_records)
    evidence_inventory = {
        "real_failures_observed": metrics["real_failures_observed"],
        "shadow_decisions": metrics["shadow_decisions"],
        "real_actions_executed": metrics["real_actions_executed"],
        "observational_recoveries": metrics["observational_recoveries"],
        "attributed_intervention_recoveries": metrics[
            "attributed_intervention_recoveries"
        ],
        "training_eligible_labels": metrics["training_eligible_labels"],
        "legacy_untracked_decisions": sum(
            decision.id not in tracked_decisions for decision in decisions
        ),
        "note": (
            "Shadow cases are not training examples. Independent captures "
            "are observational_recovery, not attributed_intervention_recovery."
        ),
    }
    summary = {
        "version": 1,
        "phase": 14,
        "scope": "real_outcome_observation_and_coverage_gate_without_execution",
        "generated_at": datetime.now(UTC).isoformat(),
        "metrics": metrics,
        "evidence_inventory": evidence_inventory,
        "legacy_untracked_decisions": evidence_inventory[
            "legacy_untracked_decisions"
        ],
        "training_readiness": {
            "minimum_rows": training_config[
                "minimum_rows_for_retraining"
            ],
            "eligible_labels": metrics["training_eligible_labels"],
            "eligible_action_counts": action_counts,
            "eligible_outcome_class_counts": class_counts,
            "coverage_passed": coverage_passed,
            "evidence_sufficient_for_experiment": evidence_sufficient,
            "retraining_stage": gate["stage"],
            "retraining_allowed": False,
            "reason": (
                "Phase 14C/14D observe real outcomes and measure coverage; "
                "Phase 15 stays blocked until the coverage gate authorizes."
            ),
        },
        "phase15_gate": {
            "candidate_ready": evidence_report["candidate_ready"],
            "authorized": False,
            "checks": evidence_report["checks"],
            "blocked_reason": phase15["blocked_reason"],
        },
        "evidence_boundaries": {
            "real_shadow_outcomes_are_causal_uplift": False,
            "natural_recovery_tracked_separately": True,
            "synthetic_and_real_data_mixed": False,
            "no_recovery_observed_means_definite_nonrecovery": False,
            "timeout_labeled_recovered_false": False,
            "observational_equals_attributed": False,
            "dataset_split": dataset_split,
        },
        "safety": {
            "execution_mode": "shadow",
            "controlled_execution_authorized": False,
            "provider_actions_enabled": False,
        },
    }
    summary_path = REPORT_DIR / "phase14_summary.json"
    _write_json(summary_path, summary)
    input_paths = [
        REPO_ROOT / "ml" / "config" / "real_outcome_schema.yaml",
        REPO_ROOT / "ml" / "config" / "outcome_observation.yaml",
        REPO_ROOT / "ml" / "src" / "record_recovery_outcome.py",
        REPO_ROOT / "ml" / "src" / "observe_recovery.py",
        REPO_ROOT / "ml" / "src" / "build_real_outcome_dataset.py",
        REPO_ROOT / "ml" / "src" / "validate_outcomes.py",
        REPO_ROOT / "ml" / "src" / "evidence_coverage.py",
        BACKEND_ROOT / "migrations" / "phase14_real_outcomes.sql",
        BACKEND_ROOT / "app" / "models" / "intervention_outcome.py",
        BACKEND_ROOT / "app" / "models" / "outcome_observation.py",
        BACKEND_ROOT / "app" / "models" / "intervention.py",
        BACKEND_ROOT / "app" / "models" / "__init__.py",
        BACKEND_ROOT / "app" / "services" / "outcome_state_machine.py",
        BACKEND_ROOT / "app" / "services" / "outcome_observation.py",
        BACKEND_ROOT / "app" / "services" / "outcome_events.py",
        BACKEND_ROOT / "app" / "services" / "webhook_service.py",
        BACKEND_ROOT / "app" / "services" / "recovery_engine.py",
        BACKEND_ROOT / "app" / "workers" / "outcome_tasks.py",
        BACKEND_ROOT / "app" / "workers" / "outcome_observer.py",
        BACKEND_ROOT / "app" / "workers" / "reliability_tasks.py",
        BACKEND_ROOT / "app" / "api" / "outcomes.py",
        BACKEND_ROOT / "app" / "database" / "connection.py",
        BACKEND_ROOT / "app" / "main.py",
        Path(__file__),
    ]
    coverage_json = REPORT_DIR / "coverage.json"
    manifest = {
        "version": 1,
        "phase": 14,
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): {
                "sha256": _sha256(path)
            }
            for path in input_paths
        },
        "artifacts": {
            str(path.relative_to(REPO_ROOT)).replace("\\", "/"): {
                "sha256": _sha256(path)
            }
            for path in (
                cases_path,
                summary_path,
                validation_path,
                coverage_path,
                coverage_json,
                REPO_ROOT / "ml" / "reports" / "phase15" / "phase15_readiness.json",
            )
            if path.exists()
        },
        "controlled_execution_authorized": False,
        "retraining_authorized": False,
        "phase15_authorized": False,
    }
    _write_json(REPORT_DIR / "phase14_manifest.json", manifest)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
