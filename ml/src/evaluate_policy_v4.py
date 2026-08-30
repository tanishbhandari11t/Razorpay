from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.analyze_support import (
    DEFAULT_ARTIFACT_DIR,
    DEFAULT_DATASET_PATH,
    build_candidate_probabilities,
)
from ml.src.failure_classifier import load_intervention_policy
from ml.src.model_pipeline import (
    DEFAULT_MANIFEST_PATH,
    load_manifest,
    validate_dataset,
)
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    decide_recovery_action,
)
from ml.src.policies.recovery_policy_v4 import (
    decide_recovery_action_v4,
    load_policy_v4,
)
from ml.src.policies.stopping_rules import RecoveryPolicyContext
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    build_context_support_table,
    load_support_policy_config,
)
from ml.src.policy_engine_simulator import (
    DEFAULT_OUTCOMES_PATH,
    DEFAULT_PHASE8_REPORT_DIR,
    _unit_interval,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase10"


def _metrics(
    name: str,
    actions: list[str],
    payment_ids: list[str],
    outcomes: pd.DataFrame,
    v3_config: dict[str, Any],
    v4_config: dict[str, Any],
) -> dict[str, Any]:
    lookup = outcomes.set_index(["payment_id", "intervention"])
    recovered = 0
    recovered_amount = 0
    cost = 0.0
    for payment_id, action in zip(payment_ids, actions, strict=True):
        if action != "no_action":
            potential = lookup.loc[(payment_id, action)]
            recovered += int(potential["recovered"])
            recovered_amount += int(potential["amount_recovered"])
        if action != "no_action":
            if name == "policy_v4":
                cost += float(
                    v4_config["actions"][action]["intervention_cost_inr"]
                )
            else:
                cost += float(
                    v3_config["interventions"][action]["cost_inr"]
                )
    interventions = sum(action != "no_action" for action in actions)
    return {
        "recovered_payments": recovered,
        "recovery_rate": round(recovered / len(actions), 8),
        "recovered_amount": recovered_amount,
        "intervention_count": interventions,
        "intervention_cost": round(cost, 2),
        "net_recovered_value": round(recovered_amount - cost, 2),
        "recovery_roi": (
            round(recovered_amount / cost, 4) if cost else None
        ),
        "selected_action_counts": dict(Counter(actions)),
    }


def evaluate_policy_v4(
    dataset_path: Path = DEFAULT_DATASET_PATH,
    outcomes_path: Path = DEFAULT_OUTCOMES_PATH,
    manifest_path: Path = DEFAULT_MANIFEST_PATH,
    artifact_dir: Path = DEFAULT_ARTIFACT_DIR,
    report_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    dataset = pd.read_csv(dataset_path)
    validate_dataset(dataset, manifest, source_path=dataset_path)
    outcomes = pd.read_csv(outcomes_path)
    support_config = load_support_policy_config()
    v3_config = load_intervention_policy()
    v4_config = load_policy_v4()
    actions = [str(action) for action in support_config["interventions"]]
    prediction_config = {"interventions": actions}
    test = (
        dataset.loc[dataset["split"].eq("test")]
        .sort_values(["prediction_time", "payment_id"])
        .reset_index(drop=True)
    )
    probabilities = build_candidate_probabilities(
        test,
        manifest,
        prediction_config,
        artifact_dir,
    )
    training = dataset.loc[dataset["split"].eq("train")]
    support_index = SupportIndex(
        build_context_support_table(training, support_config),
        support_config,
    )
    failure_lookup = (
        outcomes.drop_duplicates("payment_id")
        .set_index("payment_id")["synthetic_failure_scenario"]
    )
    seed = int(support_config["random_seed"])
    contact_rate = float(
        v3_config["simulation_assumptions"][
            "customer_contact_available_rate"
        ]
    )
    opt_out_rate = float(
        v3_config["simulation_assumptions"]["customer_opt_out_rate"]
    )
    v3_actions: list[str] = []
    v4_actions: list[str] = []
    rows: list[dict[str, Any]] = []
    v4_policy_violations = 0
    for index, row in test.iterrows():
        payment_id = str(row["payment_id"])
        customer_id = str(row["customer_id"])
        prediction_time = pd.Timestamp(row["prediction_time"]).to_pydatetime()
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=UTC)
        context = RecoveryPolicyContext(
            case_id=f"V4-{payment_id}",
            payment_id=payment_id,
            amount_inr=float(row["amount_inr"]),
            payment_status="failed",
            failure_reason=str(failure_lookup.loc[payment_id]),
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
        support = {}
        for action in actions:
            evidence = support_index.evidence(row, action)
            support[action] = CandidateSupport(
                supported=evidence.supported,
                action_count=evidence.action_count,
                effective_sample_size=evidence.effective_sample_size,
            )
        candidate_probabilities = {
            action: float(probabilities[action][index])
            for action in actions
        }
        candidate_probabilities["no_action"] = 0.0
        v3 = decide_recovery_action(
            context,
            {
                action: candidate_probabilities[action]
                for action in actions
            },
            support,
        )
        v4 = decide_recovery_action_v4(
            context,
            candidate_probabilities,
            support,
            config=v4_config,
        )
        v3_action = v3.selected_action or "no_action"
        v3_actions.append(v3_action)
        v4_actions.append(v4.selected_action)
        selected_v4 = v4.candidates[v4.selected_action]
        v4_policy_violations += int(
            v4.selected_action != "no_action"
            and not selected_v4.eligible
            and v4.failure_class != "fraud_risk"
        )
        rows.append(
            {
                "payment_id": payment_id,
                "amount_inr": float(row["amount_inr"]),
                "failure_class": v4.failure_class,
                "v3_action": v3_action,
                "v4_action": v4.selected_action,
                "no_action_probability": candidate_probabilities[
                    "no_action"
                ],
                "v4_selected_probability": selected_v4.probability,
                "v4_incremental_net_value_inr": (
                    selected_v4.incremental_net_value_inr
                ),
                "v4_reasons": json.dumps(v4.reasons),
            }
        )

    payment_ids = test["payment_id"].astype(str).tolist()
    always_retry = ["retry_payment"] * len(test)
    report = {
        "version": 1,
        "scope": (
            "One-step synthetic comparison on the frozen Phase 6 test set; "
            "no-action is conservatively assigned zero realized recovery "
            "because no counterfactual no-action outcome exists"
        ),
        "frozen_test_rows": len(test),
        "cost_disclosure": (
            "All action costs are synthetic estimates, not measured "
            "Razorpay or provider pricing."
        ),
        "always_retry": _metrics(
            "always_retry",
            always_retry,
            payment_ids,
            outcomes,
            v3_config,
            v4_config,
        ),
        "policy_v3_one_step": _metrics(
            "policy_v3",
            v3_actions,
            payment_ids,
            outcomes,
            v3_config,
            v4_config,
        ),
        "policy_v4": _metrics(
            "policy_v4",
            v4_actions,
            payment_ids,
            outcomes,
            v3_config,
            v4_config,
        ),
        "policy_v4_safety": {
            "policy_violations": v4_policy_violations,
            "execution_authorized": False,
            "qwen_used": False,
            "provider_calls": 0,
        },
        "frozen_phase8_v3": json.loads(
            (
                DEFAULT_PHASE8_REPORT_DIR / "policy_engine_metrics.json"
            ).read_text(encoding="utf-8")
        )["recovery_policy_v3"],
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "policy_v4_comparison.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(rows).to_csv(
        report_dir / "policy_v4_decisions.csv",
        index=False,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    args = parser.parse_args()
    report = evaluate_policy_v4(report_dir=args.report_dir)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
