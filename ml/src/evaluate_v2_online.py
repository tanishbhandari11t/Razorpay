from __future__ import annotations

import json
from collections import Counter
from datetime import UTC
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import yaml
from xgboost import XGBClassifier

from ml.src.evaluate_policy_sensitivity import (
    perturb_intervention_probabilities,
)
from ml.src.failure_classifier import load_intervention_policy
from ml.src.model_pipeline import (
    feature_columns,
    load_manifest,
    load_preprocessor,
    predict_probabilities,
    validate_dataset,
)
from ml.src.offline_policy_eval import (
    _deterministic_policy_arrays,
    _evaluate_policy,
    _logging_policy_arrays,
)
from ml.src.policies.recovery_policy import (
    CandidateSupport,
    decide_recovery_action,
    load_action_matrix,
)
from ml.src.policies.decision_margin import apply_decision_margin
from ml.src.policies.stopping_rules import RecoveryPolicyContext
from ml.src.policies.support_safe_policy import (
    SupportIndex,
    build_context_support_table,
    load_support_policy_config,
)
from ml.src.policy_engine_simulator import DEFAULT_OUTCOMES_PATH, _unit_interval
from ml.src.simulate_recovery import load_config as load_simulator_config


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = (
    REPO_ROOT / "ml" / "data" / "processed" / "online_training_features.csv"
)
MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "dataset_manifest_v2_online.yaml"
)
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "v2_online"
EVALUATION_CONFIG_PATH = (
    REPO_ROOT / "ml" / "config" / "phase13_evaluation.yaml"
)
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase13"
V1_METRICS_PATH = REPO_ROOT / "ml" / "artifacts" / "metrics_v1.json"
V1_V3_DECISIONS_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase10" / "policy_v4_decisions.csv"
)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _load_v2_bundle() -> tuple[Any, XGBClassifier, Any]:
    preprocessor = load_preprocessor(
        ARTIFACT_DIR / "preprocessing_v2_online.joblib"
    )
    calibrator = joblib.load(
        ARTIFACT_DIR / "calibration_v2_online.joblib"
    )
    model = XGBClassifier()
    model.load_model(ARTIFACT_DIR / "recovery_model_v2_online.json")
    return preprocessor, model, calibrator


def build_v2_candidate_probabilities(
    dataframe: pd.DataFrame,
    manifest: dict[str, Any],
    actions: list[str],
) -> dict[str, np.ndarray]:
    preprocessor, model, calibrator = _load_v2_bundle()
    base = dataframe[feature_columns(manifest)].copy()
    result = {}
    for action in actions:
        candidates = base.copy()
        candidates["chosen_intervention"] = action
        result[action] = predict_probabilities(
            preprocessor,
            model,
            candidates,
            calibrator,
        )
    return result


def _economic_metrics(
    actions: list[str],
    payment_ids: list[str],
    outcomes: pd.DataFrame,
    policy: dict[str, Any],
) -> dict[str, Any]:
    lookup = outcomes.set_index(["payment_id", "intervention"])
    recovered = 0
    recovered_amount = 0
    cost = 0.0
    for payment_id, action in zip(payment_ids, actions, strict=True):
        if action == "no_action":
            continue
        potential = lookup.loc[(payment_id, action)]
        recovered += int(potential["recovered"])
        recovered_amount += int(potential["amount_recovered"])
        cost += float(policy["interventions"][action]["cost_inr"])
    interventions = sum(action != "no_action" for action in actions)
    return {
        "recovered_payments": recovered,
        "recovery_rate": round(recovered / len(actions), 8),
        "recovered_amount_inr": recovered_amount,
        "intervention_count": interventions,
        "intervention_cost_inr": round(cost, 2),
        "net_recovered_value_inr": round(recovered_amount - cost, 2),
        "recovery_roi": (
            round(recovered_amount / cost, 4) if cost else None
        ),
        "selected_action_counts": dict(Counter(actions)),
    }


def _model_comparison() -> dict[str, Any]:
    v1 = json.loads(V1_METRICS_PATH.read_text(encoding="utf-8"))
    v2 = json.loads(
        (ARTIFACT_DIR / "metrics_v2_online.json").read_text(encoding="utf-8")
    )
    v1_test = v1["test_after_calibration"]
    v2_test = v2["test_after_calibration"]
    deltas = {
        metric: round(float(v2_test[metric]) - float(v1_test[metric]), 8)
        for metric in (
            "roc_auc",
            "pr_auc",
            "brier_score",
            "log_loss",
            "mean_absolute_calibration_error",
        )
    }
    return {
        "version": 1,
        "evaluation_split": "same_frozen_test_membership",
        "rows": int(v2_test["rows"]),
        "v1_research_model": v1_test,
        "v2_online_model": v2_test,
        "delta_v2_minus_v1": deltas,
        "interpretation": (
            "A deployability trade-off comparison on synthetic logged data; "
            "not evidence of real Razorpay recovery uplift."
        ),
    }


def evaluate_v2_online() -> dict[str, Any]:
    evaluation = yaml.safe_load(
        EVALUATION_CONFIG_PATH.read_text(encoding="utf-8")
    )
    support_config = load_support_policy_config()
    policy = load_intervention_policy()
    matrix = load_action_matrix()
    manifest = load_manifest(MANIFEST_PATH)
    dataset = pd.read_csv(DATASET_PATH)
    validate_dataset(dataset, manifest, source_path=DATASET_PATH)
    outcomes = pd.read_csv(DEFAULT_OUTCOMES_PATH)
    actions = [str(action) for action in evaluation["interventions"]]
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
    probabilities = build_v2_candidate_probabilities(test, manifest, actions)
    failure_lookup = (
        outcomes.drop_duplicates("payment_id")
        .set_index("payment_id")["synthetic_failure_scenario"]
    )
    seed = int(support_config["random_seed"])
    contact_rate = float(
        policy["simulation_assumptions"]["customer_contact_available_rate"]
    )
    opt_out_rate = float(
        policy["simulation_assumptions"]["customer_opt_out_rate"]
    )
    rows = []
    selected_actions: list[str] = []
    policy_violations = 0
    for index, row in test.iterrows():
        payment_id = str(row["payment_id"])
        customer_id = str(row["customer_id"])
        prediction_time = pd.Timestamp(row["prediction_time"]).to_pydatetime()
        if prediction_time.tzinfo is None:
            prediction_time = prediction_time.replace(tzinfo=UTC)
        context = RecoveryPolicyContext(
            case_id=f"V2-ONLINE-{payment_id}",
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
        decision = decide_recovery_action(
            context,
            candidate_probabilities,
            support,
            policy=policy,
            action_matrix=matrix,
        )
        margin_gate = apply_decision_margin(
            decision,
            fallback_action=str(
                support_config["decision"]["preferred_fallback_action"]
            ),
            threshold_inr=float(
                evaluation["decision_margins"][
                    "minimum_expected_value_margin_inr"
                ]
            ),
        )
        selected = margin_gate.selected_action or "no_action"
        selected_actions.append(selected)
        policy_violations += int(
            selected != "no_action" and not decision.risk_checks_passed
        )
        ranked_probabilities = sorted(
            candidate_probabilities.values(),
            reverse=True,
        )
        probability_margin = (
            ranked_probabilities[0] - ranked_probabilities[1]
        )
        output = {
            "payment_id": payment_id,
            "customer_id": customer_id,
            "prediction_time": row["prediction_time"],
            "logged_action": row["chosen_intervention"],
            "logging_probability": float(row["policy_probability"]),
            "logged_recovered": int(row["recovered"]),
            "base_policy_intervention": row["base_policy_intervention"],
            "fraud_flag": int(row["fraud_flag"]),
            "selected_action": selected,
            "decision_type": decision.decision_type.value,
            "fallback_used": (
                decision.fallback_used or margin_gate.fallback_triggered
            ),
            "failure_class": decision.failure_class,
            "risk_checks_passed": decision.risk_checks_passed,
            "probability_margin": probability_margin,
            "expected_value_margin_inr": (
                margin_gate.decision_margin_inr
            ),
            "margin_fallback_triggered": margin_gate.fallback_triggered,
            "decision_reasons": json.dumps(
                [*decision.reasons, margin_gate.reason]
            ),
        }
        for action in actions:
            output[f"predicted_{action}_probability"] = (
                candidate_probabilities[action]
            )
            output[f"{action}_supported"] = support[action].supported
            output[f"{action}_support_count"] = support[action].action_count
            output[f"{action}_support_ess"] = (
                support[action].effective_sample_size
            )
        rows.append(output)
    decisions = pd.DataFrame(rows)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    decisions_path = REPORT_DIR / "v2_online_policy_decisions.csv"
    decisions.to_csv(decisions_path, index=False, float_format="%.8f")

    model_comparison = _model_comparison()
    _write_json(
        REPORT_DIR / "v2_online_model_comparison.json",
        model_comparison,
    )

    v1_v3 = pd.read_csv(V1_V3_DECISIONS_PATH)[
        ["payment_id", "v3_action"]
    ]
    decisions = decisions.merge(
        v1_v3,
        on="payment_id",
        how="left",
        validate="one_to_one",
    )
    if decisions["v3_action"].isna().any():
        raise ValueError("Frozen V1/V3 decisions do not cover V2 test rows")
    always_retry = pd.Series(
        np.where(decisions["fraud_flag"].eq(1), "no_action", "retry_payment"),
        dtype="string",
    )
    policy_arrays = {
        "always_retry": _deterministic_policy_arrays(
            decisions,
            always_retry,
            actions,
        ),
        "historical_base_policy": _deterministic_policy_arrays(
            decisions,
            decisions["base_policy_intervention"].astype("string"),
            actions,
        ),
        "logging_policy": _logging_policy_arrays(decisions, actions),
        "frozen_v1_v3_one_step": _deterministic_policy_arrays(
            decisions,
            decisions["v3_action"].astype("string"),
            actions,
        ),
        "v2_online_with_frozen_v3": _deterministic_policy_arrays(
            decisions,
            decisions["selected_action"].astype("string"),
            actions,
        ),
    }
    ope_config = {
        "interventions": actions,
        "random_seed": int(support_config["random_seed"]),
        "offline_evaluation": evaluation["offline_evaluation"],
    }
    minimum_propensity = 1 / float(
        evaluation["offline_evaluation"]["maximum_importance_weight"]
    )
    ope_policies = {
        name: _evaluate_policy(
            name,
            arrays,
            decisions,
            minimum_propensity,
            ope_config,
        )
        for name, arrays in policy_arrays.items()
    }
    clipping = {}
    for threshold in evaluation["offline_evaluation"][
        "clipping_sensitivity"
    ]:
        clipping[str(threshold)] = {
            name: {
                key: value
                for key, value in _evaluate_policy(
                    name,
                    arrays,
                    decisions,
                    float(threshold),
                    {
                        **ope_config,
                        "offline_evaluation": {
                            **ope_config["offline_evaluation"],
                            "bootstrap_iterations": 1,
                        },
                    },
                ).items()
                if key
                in {
                    "ips",
                    "self_normalized_ips",
                    "doubly_robust",
                    "importance_weight_effective_sample_size",
                    "maximum_importance_weight",
                }
            }
            for name, arrays in policy_arrays.items()
        }
    ope_report = {
        "version": 1,
        "evaluation_split": "frozen_test",
        "rows": len(decisions),
        "minimum_propensity": minimum_propensity,
        "bootstrap": evaluation["offline_evaluation"],
        "policies": ope_policies,
        "clipping_sensitivity": clipping,
        "counterfactual_outcomes_used": False,
        "real_shadow_cases_used": False,
        "interpretation": (
            "Logged-observed OPE with synthetic logging propensities. "
            "Overlapping intervals do not support an improvement claim."
        ),
    }
    _write_json(REPORT_DIR / "v2_online_ope.json", ope_report)

    payment_ids = decisions["payment_id"].astype(str).tolist()
    economics = {
        "version": 1,
        "scope": "one_step_synthetic_counterfactual",
        "cost_disclosure": (
            "Action costs are synthetic operational assumptions, not "
            "measured Razorpay pricing."
        ),
        "always_retry": _economic_metrics(
            always_retry.tolist(),
            payment_ids,
            outcomes,
            policy,
        ),
        "frozen_v1_v3_one_step": _economic_metrics(
            decisions["v3_action"].astype(str).tolist(),
            payment_ids,
            outcomes,
            policy,
        ),
        "v2_online_with_frozen_v3": _economic_metrics(
            decisions["selected_action"].astype(str).tolist(),
            payment_ids,
            outcomes,
            policy,
        ),
        "policy_safety": {
            "policy_violations": policy_violations,
            "fraud_automated_actions": int(
                decisions.loc[
                    decisions["fraud_flag"].eq(1),
                    "selected_action",
                ].isin(
                    [
                        "retry_payment",
                        "payment_link",
                        "whatsapp_reminder",
                    ]
                ).sum()
            ),
            "provider_calls": 0,
            "execution_authorized": False,
        },
        "margin_evaluation": {
            "v2_validation_calibration_margin": evaluation[
                "decision_margins"
            ]["v2_validation_calibration_margin"],
            "weak_probability_margin_count": int(
                decisions["probability_margin"].lt(
                    float(
                        evaluation["decision_margins"][
                            "v2_validation_calibration_margin"
                        ]
                    )
                ).sum()
            ),
            "weak_probability_margin_rate": round(
                float(
                    decisions["probability_margin"].lt(
                        float(
                            evaluation["decision_margins"][
                                "v2_validation_calibration_margin"
                            ]
                        )
                    ).mean()
                ),
                8,
            ),
            "minimum_expected_value_margin_inr": evaluation[
                "decision_margins"
            ]["minimum_expected_value_margin_inr"],
            "weak_expected_value_margin_count": int(
                decisions["expected_value_margin_inr"]
                .notna()
                .mul(
                    decisions["expected_value_margin_inr"].lt(
                        float(
                            evaluation["decision_margins"][
                                "minimum_expected_value_margin_inr"
                            ]
                        )
                    )
                )
                .sum()
            ),
            "margin_fallback_count": int(
                decisions["margin_fallback_triggered"].sum()
            ),
            "fallback_count": int(decisions["fallback_used"].sum()),
            "fallback_rate": round(
                float(decisions["fallback_used"].mean()),
                8,
            ),
        },
    }
    _write_json(
        REPORT_DIR / "v2_online_policy_economics.json",
        economics,
    )

    simulator_config = load_simulator_config()
    sensitivity_rows = []
    for intervention in actions:
        for multiplier in evaluation["sensitivity"][
            "intervention_probability_multipliers"
        ]:
            perturbed = perturb_intervention_probabilities(
                outcomes,
                intervention=intervention,
                multiplier=float(multiplier),
                seed=int(simulator_config["seed"]),
            )
            retry_metrics = _economic_metrics(
                always_retry.tolist(),
                payment_ids,
                perturbed,
                policy,
            )
            v2_metrics = _economic_metrics(
                decisions["selected_action"].astype(str).tolist(),
                payment_ids,
                perturbed,
                policy,
            )
            sensitivity_rows.append(
                {
                    "perturbed_intervention": intervention,
                    "probability_multiplier": float(multiplier),
                    "always_retry_recovery_rate": retry_metrics[
                        "recovery_rate"
                    ],
                    "v2_online_recovery_rate": v2_metrics["recovery_rate"],
                    "v2_online_beats_always_retry": (
                        float(v2_metrics["recovery_rate"])
                        > float(retry_metrics["recovery_rate"])
                    ),
                }
            )
    wins = sum(
        row["v2_online_beats_always_retry"] for row in sensitivity_rows
    )
    sensitivity_report = {
        "version": 1,
        "scope": "synthetic_counterfactual_sensitivity",
        "scenarios": sensitivity_rows,
        "scenarios_won": wins,
        "scenarios_total": len(sensitivity_rows),
        "robust_win_minimum": evaluation["sensitivity"][
            "robust_win_minimum_scenarios"
        ],
        "robust_uplift_supported": wins
        >= int(
            evaluation["sensitivity"]["robust_win_minimum_scenarios"]
        ),
        "real_uplift_claim_allowed": False,
    }
    _write_json(
        REPORT_DIR / "v2_online_sensitivity.json",
        sensitivity_report,
    )
    return {
        "model": model_comparison,
        "ope": ope_report,
        "economics": economics,
        "sensitivity": sensitivity_report,
    }


if __name__ == "__main__":
    result = evaluate_v2_online()
    print(
        json.dumps(
            {
                "model_delta": result["model"]["delta_v2_minus_v1"],
                "economics": result["economics"][
                    "v2_online_with_frozen_v3"
                ],
                "sensitivity": {
                    "won": result["sensitivity"]["scenarios_won"],
                    "total": result["sensitivity"]["scenarios_total"],
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
