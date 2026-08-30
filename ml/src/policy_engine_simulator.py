from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ml.src.analyze_support import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET_PATH,
    DEFAULT_REPORT_DIR,
    build_candidate_probabilities,
)
from ml.src.failure_classifier import load_intervention_policy
from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    file_sha256,
    load_manifest,
    membership_sha256,
    validate_dataset,
)
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    DecisionType,
    decide_recovery_action,
    load_action_matrix,
)
from ml.src.policies.stopping_rules import (
    InterventionAttempt,
    RecoveryPolicyContext,
)
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    build_context_support_table,
    load_support_policy_config,
)
from ml.src.recovery_state_machine import RecoveryState, RecoveryStateMachine


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTCOMES_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "intervention_outcomes.csv"
)
DEFAULT_V1_DECISIONS_PATH = (
    REPO_ROOT / "ml" / "data" / "evaluation" / "test_policy_decisions_v1.csv"
)
DEFAULT_PHASE8_REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase8"


def _unit_interval(seed: int, *parts: object) -> float:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / 2**64


def _single_action_metrics(
    test: pd.DataFrame,
    outcomes: pd.DataFrame,
    actions: pd.Series,
    policy_config: dict[str, Any],
) -> dict[str, float | int]:
    decisions = pd.DataFrame(
        {
            "payment_id": test["payment_id"].to_numpy(),
            "action": actions.to_numpy(),
        }
    )
    automated = decisions.loc[decisions["action"].ne("no_action")]
    selected = automated.merge(
        outcomes,
        left_on=["payment_id", "action"],
        right_on=["payment_id", "intervention"],
        validate="one_to_one",
    )
    recovered_amount = int(selected["amount_recovered"].sum())
    cost = sum(
        float(policy_config["interventions"][action]["cost_inr"])
        for action in automated["action"]
    )
    recoveries = int(selected["recovered"].sum())
    return {
        "recovered_payments": recoveries,
        "recovery_rate": round(recoveries / len(test), 8),
        "recovered_amount": recovered_amount,
        "intervention_count": len(automated),
        "intervention_cost": round(cost, 2),
        "recovered_amount_per_intervention": round(
            recovered_amount / len(automated),
            2,
        )
        if len(automated)
        else 0.0,
        "recovery_roi": round(recovered_amount / cost, 4) if cost else None,
    }


def simulate_policy_engine(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    phase7_report_dir: Path = DEFAULT_REPORT_DIR,
    report_dir: Path = DEFAULT_PHASE8_REPORT_DIR,
    v1_decisions_path: Path = DEFAULT_V1_DECISIONS_PATH,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)
    outcomes = pd.read_csv(outcomes_path)
    support_config = load_support_policy_config()
    policy_config = load_intervention_policy()
    matrix = load_action_matrix()
    simulator_policy = deepcopy(policy_config)
    simulator_max_attempts = int(
        policy_config["simulation_assumptions"][
            "max_attempts_per_action"
        ]
    )
    for action in simulator_policy["interventions"].values():
        action["max_attempts"] = min(
            int(action["max_attempts"]),
            simulator_max_attempts,
        )

    training = dataset.loc[dataset["split"].eq("train")]
    support_index = SupportIndex(
        build_context_support_table(training, support_config),
        support_config,
    )
    test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values(["prediction_time", "payment_id"])
        .reset_index(drop=True)
    )
    candidate_probabilities = build_candidate_probabilities(
        test,
        manifest,
        support_config,
        artifact_dir,
    )
    scenario_lookup = (
        outcomes.drop_duplicates("payment_id")
        .set_index("payment_id")["synthetic_failure_scenario"]
    )
    outcome_lookup = outcomes.set_index(["payment_id", "intervention"])

    decision_rows: list[dict[str, Any]] = []
    case_rows: list[dict[str, Any]] = []
    seed = int(support_config["random_seed"])
    contact_rate = float(
        policy_config["simulation_assumptions"][
            "customer_contact_available_rate"
        ]
    )
    opt_out_rate = float(
        policy_config["simulation_assumptions"]["customer_opt_out_rate"]
    )

    for index, row in test.iterrows():
        payment_id = str(row["payment_id"])
        customer_id = str(row["customer_id"])
        prediction_time = pd.Timestamp(row["prediction_time"]).to_pydatetime()
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=UTC)
        context = RecoveryPolicyContext(
            case_id=f"SIM-{payment_id}",
            payment_id=payment_id,
            amount_inr=float(row["amount_inr"]),
            payment_status="failed",
            failure_reason=str(scenario_lookup.loc[payment_id]),
            fraud_flag=int(row["fraud_flag"]),
            case_created_at=prediction_time,
            now=prediction_time,
            customer_contact_available=(
                _unit_interval(seed, "contact", customer_id) < contact_rate
            ),
            customer_opted_out=(
                _unit_interval(seed, "opt-out", customer_id) < opt_out_rate
            ),
            valid_payment_context=True,
        )
        probabilities = {
            action: float(candidate_probabilities[action][index])
            for action in support_config["interventions"]
        }
        support = {}
        for action in support_config["interventions"]:
            evidence = support_index.evidence(row, action)
            support[action] = CandidateSupport(
                supported=evidence.supported,
                action_count=evidence.action_count,
                effective_sample_size=evidence.effective_sample_size,
            )

        machine = RecoveryStateMachine()
        recovered = False
        recovered_amount = 0
        total_cost = 0.0
        stopped_reason: str | None = None
        while machine.state not in {
            RecoveryState.RECOVERED,
            RecoveryState.STOPPED,
            RecoveryState.ESCALATED,
        }:
            if machine.state in {
                RecoveryState.AT_RISK,
                RecoveryState.NEXT_ACTION,
            }:
                machine.transition(RecoveryState.DIAGNOSED)
            decision = decide_recovery_action(
                context,
                probabilities,
                support,
                policy=simulator_policy,
                action_matrix=matrix,
            )
            decision_row = {
                "payment_id": payment_id,
                "customer_id": customer_id,
                "decision_number": len(context.attempts) + 1,
                "decision_type": decision.decision_type.value,
                "selected_action": decision.selected_action,
                "failure_class": decision.failure_class,
                "reasons": json.dumps(decision.reasons),
                "probabilities": json.dumps(
                    decision.probabilities,
                    sort_keys=True,
                ),
                "expected_values": json.dumps(
                    decision.expected_values,
                    sort_keys=True,
                ),
                "fallback_used": decision.fallback_used,
                "risk_checks_passed": decision.risk_checks_passed,
                "policy_version": decision.policy_version,
                "dry_run": decision.dry_run,
            }
            decision_rows.append(decision_row)
            if decision.selected_action is None:
                stopped_reason = ",".join(decision.reasons)
                machine.transition(RecoveryState.STOPPED)
                break

            machine.transition(RecoveryState.ACTION_SELECTED)
            machine.transition(RecoveryState.ACTION_EXECUTED)
            action = decision.selected_action
            potential = outcome_lookup.loc[(payment_id, action)]
            action_cost = float(
                policy_config["interventions"][action]["cost_inr"]
            )
            total_cost += action_cost
            context.attempts.append(
                InterventionAttempt(
                    action=action,
                    status=(
                        "simulated_recovered"
                        if int(potential["recovered"]) == 1
                        else "simulated_failed"
                    ),
                    executed_at=context.now,
                    cost_inr=action_cost,
                )
            )
            if int(potential["recovered"]) == 1:
                recovered = True
                recovered_amount = int(potential["amount_recovered"])
                context.payment_status = "recovered"
                machine.transition(RecoveryState.RECOVERED)
                break
            if action == "escalate_to_merchant":
                stopped_reason = "manual_escalation_completed"
                machine.transition(RecoveryState.ESCALATED)
                break
            machine.transition(RecoveryState.FAILED)
            cooldown = float(
                policy_config["interventions"][action]["cooldown_hours"]
            )
            context.now = context.now + timedelta(hours=cooldown)
            machine.transition(RecoveryState.NEXT_ACTION)

        case_rows.append(
            {
                "payment_id": payment_id,
                "customer_id": customer_id,
                "recovered": int(recovered),
                "amount_inr": int(row["amount_inr"]),
                "amount_recovered": recovered_amount,
                "intervention_count": len(context.attempts),
                "intervention_cost": round(total_cost, 2),
                "terminal_state": machine.state.value,
                "stopped_reason": stopped_reason,
                "fraud_flag": int(row["fraud_flag"]),
                "customer_opted_out": context.customer_opted_out,
                "contact_available": context.customer_contact_available,
                "state_history": json.dumps(
                    [state.value for state in machine.history]
                ),
            }
        )

    decisions = pd.DataFrame(decision_rows)
    cases = pd.DataFrame(case_rows)
    v1_decisions = pd.read_csv(v1_decisions_path).set_index("payment_id")
    v2_decisions = pd.read_csv(
        phase7_report_dir / "policy_decisions.csv"
    ).set_index("payment_id")
    fraud = test["fraud_flag"].eq(1)
    always_retry = pd.Series(
        np.where(fraud, "no_action", "retry_payment"),
        dtype="string",
    )
    historical = test["base_policy_intervention"].astype("string")
    logged = test["chosen_intervention"].astype("string")
    v1 = test["payment_id"].map(
        v1_decisions["recoverai_intervention"]
    ).astype("string")
    v2 = test["payment_id"].map(v2_decisions["selected_action"]).astype(
        "string"
    )
    recovered_amount = int(cases["amount_recovered"].sum())
    intervention_count = int(cases["intervention_count"].sum())
    intervention_cost = float(cases["intervention_cost"].sum())
    v3_metrics = {
        "recovered_payments": int(cases["recovered"].sum()),
        "recovery_rate": round(float(cases["recovered"].mean()), 8),
        "recovered_amount": recovered_amount,
        "revenue_at_risk": int(cases["amount_inr"].sum()),
        "intervention_count": intervention_count,
        "actions_per_case": round(intervention_count / len(cases), 8),
        "intervention_cost": round(intervention_cost, 2),
        "recovery_roi": round(recovered_amount / intervention_cost, 4)
        if intervention_cost
        else None,
        "recovered_amount_per_intervention": round(
            recovered_amount / intervention_count,
            2,
        )
        if intervention_count
        else 0.0,
        "fallback_decisions": int(decisions["fallback_used"].sum()),
        "blocked_decisions": int(
            decisions["decision_type"].eq(DecisionType.BLOCK.value).sum()
        ),
        "stop_decisions": int(
            decisions["decision_type"].eq(DecisionType.STOP.value).sum()
        ),
        "stop_rate": round(
            float(cases["terminal_state"].eq("stopped").mean()),
            8,
        ),
        "fraud_cases": int(cases["fraud_flag"].sum()),
        "fraud_automated_actions": int(
            decisions.loc[
                decisions["payment_id"].isin(
                    cases.loc[cases["fraud_flag"].eq(1), "payment_id"]
                ),
                "selected_action",
            ].isin(
                [
                    "retry_payment",
                    "payment_link",
                    "whatsapp_reminder",
                ]
            ).sum()
        ),
        "policy_violations": int(
            (
                decisions["selected_action"].notna()
                & ~decisions["risk_checks_passed"]
            ).sum()
        ),
        "selected_action_counts": {
            str(key): int(value)
            for key, value in decisions["selected_action"]
            .fillna("no_action")
            .value_counts()
            .items()
        },
        "terminal_state_counts": {
            str(key): int(value)
            for key, value in cases["terminal_state"].value_counts().items()
        },
    }
    report = {
        "version": 1,
        "policy_version": policy_config["policy_version"],
        "scope": (
            "Dry-run policy-engine simulation using frozen calibrated V1 "
            "predictions and synthetic counterfactual outcomes"
        ),
        "frozen_test_rows": len(test),
        "simulator_max_attempts_per_action": simulator_max_attempts,
        "baselines": {
            "always_retry": _single_action_metrics(
                test,
                outcomes,
                always_retry,
                policy_config,
            ),
            "historical_base_policy": _single_action_metrics(
                test,
                outcomes,
                historical,
                policy_config,
            ),
            "logging_policy_observed": _single_action_metrics(
                test,
                outcomes,
                logged,
                policy_config,
            ),
            "recoverai_v1": _single_action_metrics(
                test,
                outcomes,
                v1,
                policy_config,
            ),
            "support_safe_v2": _single_action_metrics(
                test,
                outcomes,
                v2,
                policy_config,
            ),
        },
        "recovery_policy_v3": v3_metrics,
        "execution_mode": "dry_run",
        "razorpay_api_calls": 0,
        "qwen_used": False,
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    decisions_path = report_dir / "policy_engine_decisions.csv"
    cases_path = report_dir / "policy_engine_cases.csv"
    metrics_path = report_dir / "policy_engine_metrics.json"
    decisions.to_csv(decisions_path, index=False, float_format="%.8f")
    cases.to_csv(cases_path, index=False, float_format="%.8f")
    metrics_path.write_text(
        json.dumps(report, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    phase8_manifest = {
        "version": 1,
        "policy_version": policy_config["policy_version"],
        "frozen_test_membership_sha256": membership_sha256(
            test["payment_id"]
        ),
        "intervention_policy_sha256": file_sha256(
            REPO_ROOT / "ml" / "config" / "intervention_policy.yaml"
        ),
        "action_matrix_sha256": file_sha256(
            REPO_ROOT / "ml" / "config" / "action_matrix.yaml"
        ),
        "reports": {
            "policy_engine_decisions_sha256": file_sha256(decisions_path),
            "policy_engine_cases_sha256": file_sha256(cases_path),
            "policy_engine_metrics_sha256": file_sha256(metrics_path),
        },
    }
    (report_dir / "phase8_report_manifest.json").write_text(
        json.dumps(phase8_manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("=== RECOVERY POLICY ENGINE V3 ===")
    print(json.dumps(report, indent=2, sort_keys=True))
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the bounded Phase 8 recovery policy simulator."
    )
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET_PATH)
    parser.add_argument("--outcomes", type=Path, default=DEFAULT_OUTCOMES_PATH)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST_PATH)
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    parser.add_argument(
        "--phase7-reports",
        type=Path,
        default=DEFAULT_REPORT_DIR,
    )
    parser.add_argument(
        "--reports",
        type=Path,
        default=DEFAULT_PHASE8_REPORT_DIR,
    )
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    simulate_policy_engine(
        arguments.dataset,
        arguments.outcomes,
        arguments.manifest,
        arguments.artifacts,
        arguments.phase7_reports,
        arguments.reports,
    )
