from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml

from ml.src.policies.recovery_policy import (
    CandidateSupport,
    DecisionType,
    RecoveryDecision,
    decide_recovery_action,
)
from ml.src.policies.stopping_rules import RecoveryPolicyContext


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_V4_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "policy_v4.yaml"
POLICY_V3_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "config" / "policy_v3_manifest.yaml"
)
PHASE9_MANIFEST_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase9" / "phase9_report_manifest.json"
)


@dataclass(frozen=True)
class EconomicCandidate:
    action: str
    probability: float
    expected_recovered_amount_inr: float
    intervention_cost_inr: float
    risk_penalty_inr: float
    expected_net_value_inr: float
    incremental_net_value_inr: float
    incremental_recovery_to_cost_ratio: float | None
    eligible: bool
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class RecoveryDecisionV4:
    case_id: str
    payment_id: str
    decision_type: str
    selected_action: str
    reasons: tuple[str, ...]
    failure_class: str
    candidates: dict[str, EconomicCandidate]
    v3_decision: RecoveryDecision
    policy_version: str = "recovery_policy_v4"
    execution_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_policy_v4(
    path: Path = DEFAULT_V4_CONFIG_PATH,
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if (
        _sha256(POLICY_V3_MANIFEST_PATH)
        != config["frozen_v3"]["policy_v3_manifest_sha256"]
    ):
        raise ValueError("Frozen Policy V3 manifest changed")
    if (
        _sha256(PHASE9_MANIFEST_PATH)
        != config["frozen_v3"]["phase9_report_manifest_sha256"]
    ):
        raise ValueError("Frozen Phase 9 report manifest changed")
    if config["safety"]["execution_authorized"]:
        raise ValueError("Policy V4 execution must remain disabled")
    return config


def decide_recovery_action_v4(
    context: RecoveryPolicyContext,
    calibrated_probabilities: Mapping[str, float],
    support_by_action: Mapping[str, CandidateSupport],
    *,
    config: dict[str, Any] | None = None,
) -> RecoveryDecisionV4:
    policy = config or load_policy_v4()
    required = {*support_by_action, "no_action"}
    if set(calibrated_probabilities) != required:
        raise ValueError(
            "Policy V4 probabilities must cover V3 actions and no_action"
        )
    v3_probabilities = {
        action: float(calibrated_probabilities[action])
        for action in support_by_action
    }
    v3 = decide_recovery_action(
        context,
        v3_probabilities,
        support_by_action,
    )
    amount = float(context.amount_inr)
    no_action_source = str(
        policy["actions"]["no_action"]["probability_source"]
    )
    if no_action_source != "conservative_zero":
        raise ValueError("Unsupported no-action probability source")
    no_action_probability = 0.0
    no_action_value = no_action_probability * amount
    candidates: dict[str, EconomicCandidate] = {
        "no_action": EconomicCandidate(
            action="no_action",
            probability=no_action_probability,
            expected_recovered_amount_inr=no_action_value,
            intervention_cost_inr=0.0,
            risk_penalty_inr=0.0,
            expected_net_value_inr=no_action_value,
            incremental_net_value_inr=0.0,
            incremental_recovery_to_cost_ratio=None,
            eligible=True,
            reasons=(),
        )
    }
    minimum_incremental = float(
        policy["objective"]["minimum_incremental_net_value_inr"]
    )
    minimum_ratio = float(
        policy["objective"][
            "minimum_incremental_recovery_to_cost_ratio"
        ]
    )
    for action, candidate in v3.candidates.items():
        configured_cost = float(
            policy["actions"][action]["intervention_cost_inr"]
        )
        if configured_cost != candidate.cost_inr:
            raise ValueError(f"Policy cost mismatch for {action}")
        expected_recovered = candidate.probability * amount
        net_value = (
            expected_recovered
            - configured_cost
            - candidate.risk_penalty_inr
        )
        incremental_recovery = (
            candidate.probability - no_action_probability
        ) * amount
        ratio = (
            incremental_recovery / configured_cost
            if configured_cost > 0
            else None
        )
        reasons = list(candidate.eligibility_reasons)
        incremental_net = net_value - no_action_value
        if incremental_net < minimum_incremental:
            reasons.append("incremental_net_value_below_threshold")
        if ratio is not None and ratio < minimum_ratio:
            reasons.append("incremental_recovery_to_cost_below_threshold")
        candidates[action] = EconomicCandidate(
            action=action,
            probability=candidate.probability,
            expected_recovered_amount_inr=expected_recovered,
            intervention_cost_inr=configured_cost,
            risk_penalty_inr=candidate.risk_penalty_inr,
            expected_net_value_inr=net_value,
            incremental_net_value_inr=incremental_net,
            incremental_recovery_to_cost_ratio=ratio,
            eligible=candidate.eligible and not reasons,
            reasons=tuple(reasons),
        )

    fraud_action = str(policy["safety"]["fraud_action"])
    hard_stop_reasons = {
        "payment_already_recovered",
        "customer_opted_out",
        "maximum_intervention_budget_reached",
        "recovery_window_expired",
        "recent_intervention_limit_reached",
    }
    if any(reason in hard_stop_reasons for reason in v3.reasons):
        selected = "no_action"
        decision_type = DecisionType.STOP.value
        reasons = ("v3_stopping_rule_preserved", *v3.reasons)
    elif v3.failure_class == "fraud_risk":
        selected = fraud_action
        decision_type = DecisionType.FALLBACK.value
        reasons = ("fraud_manual_escalation_preserved",)
    else:
        v3_selected = (
            candidates.get(v3.selected_action)
            if v3.selected_action is not None
            else None
        )
        if v3_selected is not None and v3_selected.eligible:
            selected = v3_selected.action
            decision_type = v3.decision_type.value
            reasons = (
                "v3_action_passed_incremental_economic_gate",
            )
        else:
            selected = "no_action"
            decision_type = DecisionType.STOP.value
            reasons = ("no_action_has_better_risk_adjusted_value",)

    return RecoveryDecisionV4(
        case_id=context.case_id,
        payment_id=context.payment_id,
        decision_type=decision_type,
        selected_action=selected,
        reasons=reasons,
        failure_class=v3.failure_class,
        candidates=candidates,
        v3_decision=v3,
    )
