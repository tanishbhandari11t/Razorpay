from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, Mapping

import yaml

from ml.src.failure_classifier import (
    FailureDiagnosis,
    classify_failure,
    load_intervention_policy,
)
from ml.src.policies.stopping_rules import (
    RecoveryPolicyContext,
    action_control_reason,
    count_action_attempts,
    evaluate_stopping_rules,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_ACTION_MATRIX_PATH = (
    REPO_ROOT / "ml" / "config" / "action_matrix.yaml"
)


class DecisionType(StrEnum):
    ALLOW = "allow"
    BLOCK = "block"
    FALLBACK = "fallback"
    STOP = "stop"


@dataclass(frozen=True)
class CandidateSupport:
    supported: bool
    action_count: int = 0
    effective_sample_size: float = 0.0


@dataclass(frozen=True)
class CandidateEvaluation:
    action: str
    probability: float
    eligible: bool
    eligibility_reasons: tuple[str, ...]
    supported: bool
    support_count: int
    support_effective_sample_size: float
    cost_inr: float
    risk_penalty_inr: float
    expected_value_inr: float


@dataclass(frozen=True)
class RecoveryDecision:
    case_id: str
    payment_id: str
    decision_type: DecisionType
    selected_action: str | None
    reasons: tuple[str, ...]
    failure_class: str
    failure_rule: str
    probabilities: dict[str, float]
    expected_values: dict[str, float]
    candidates: dict[str, CandidateEvaluation]
    fallback_used: bool
    risk_checks_passed: bool
    policy_version: str
    dry_run: bool

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["decision_type"] = self.decision_type.value
        return value


def load_action_matrix(
    path: Path = DEFAULT_ACTION_MATRIX_PATH,
) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _requirement_status(
    requirement: str,
    context: RecoveryPolicyContext,
    diagnosis: FailureDiagnosis,
) -> bool:
    values = {
        "payment_failed": context.payment_status.lower() == "failed",
        "retryable_failure": diagnosis.retryable,
        "valid_payment_context": context.valid_payment_context,
        "customer_contact_available": context.customer_contact_available,
        "customer_not_opted_out": not context.customer_opted_out,
    }
    if requirement not in values:
        raise ValueError(f"Unknown action requirement: {requirement}")
    return bool(values[requirement])


def _evaluate_candidate(
    action: str,
    probability: float,
    context: RecoveryPolicyContext,
    diagnosis: FailureDiagnosis,
    support: CandidateSupport,
    policy: dict[str, Any],
    matrix: dict[str, Any],
) -> CandidateEvaluation:
    reasons: list[str] = []
    matrix_rule = matrix["failure_classes"][diagnosis.failure_class.value][action]
    if matrix_rule != "allow":
        reasons.append(f"failure_matrix:{matrix_rule}")
    for requirement in matrix["requirements"][action]:
        if not _requirement_status(requirement, context, diagnosis):
            reasons.append(f"requirement_failed:{requirement}")
    control_reason = action_control_reason(context, action, policy)
    if control_reason:
        reasons.append(control_reason)

    assumptions = policy["interventions"][action]
    support_required = bool(assumptions["support_required"])
    if support_required and not support.supported:
        reasons.append("insufficient_contextual_support")
    attempt_count = count_action_attempts(context, action)
    cost = float(assumptions["cost_inr"])
    risk_penalty = float(assumptions["base_risk_penalty_inr"]) + (
        attempt_count * float(assumptions["repeat_risk_penalty_inr"])
    )
    expected_value = float(probability) * float(context.amount_inr) - cost - risk_penalty
    return CandidateEvaluation(
        action=action,
        probability=float(probability),
        eligible=not reasons,
        eligibility_reasons=tuple(reasons),
        supported=(support.supported or not support_required),
        support_count=int(support.action_count),
        support_effective_sample_size=float(
            support.effective_sample_size
        ),
        cost_inr=cost,
        risk_penalty_inr=risk_penalty,
        expected_value_inr=expected_value,
    )


def _decision(
    *,
    context: RecoveryPolicyContext,
    policy: dict[str, Any],
    diagnosis: FailureDiagnosis,
    decision_type: DecisionType,
    selected_action: str | None,
    reasons: list[str],
    candidates: dict[str, CandidateEvaluation],
    fallback_used: bool,
    risk_checks_passed: bool,
) -> RecoveryDecision:
    return RecoveryDecision(
        case_id=context.case_id,
        payment_id=context.payment_id,
        decision_type=decision_type,
        selected_action=selected_action,
        reasons=tuple(reasons),
        failure_class=diagnosis.failure_class.value,
        failure_rule=diagnosis.matched_rule,
        probabilities={
            action: round(candidate.probability, 8)
            for action, candidate in candidates.items()
        },
        expected_values={
            action: round(candidate.expected_value_inr, 2)
            for action, candidate in candidates.items()
        },
        candidates=candidates,
        fallback_used=fallback_used,
        risk_checks_passed=risk_checks_passed,
        policy_version=str(policy["policy_version"]),
        dry_run=bool(policy["dry_run"]),
    )


def decide_recovery_action(
    context: RecoveryPolicyContext,
    calibrated_probabilities: Mapping[str, float],
    support_by_action: Mapping[str, CandidateSupport],
    *,
    policy: dict[str, Any] | None = None,
    action_matrix: dict[str, Any] | None = None,
) -> RecoveryDecision:
    policy_config = policy or load_intervention_policy()
    matrix = action_matrix or load_action_matrix()
    interventions = list(policy_config["interventions"])
    if set(calibrated_probabilities) != set(interventions):
        raise ValueError("Calibrated probabilities must cover every intervention")
    if not all(
        0 <= float(probability) <= 1
        for probability in calibrated_probabilities.values()
    ):
        raise ValueError("Calibrated probabilities must be in [0, 1]")

    diagnosis = classify_failure(
        context.failure_reason,
        fraud_flag=context.fraud_flag,
        config=policy_config,
    )
    stopping = evaluate_stopping_rules(context, policy_config)
    candidates = {
        action: _evaluate_candidate(
            action,
            float(calibrated_probabilities[action]),
            context,
            diagnosis,
            support_by_action.get(action, CandidateSupport(False)),
            policy_config,
            matrix,
        )
        for action in interventions
    }
    if stopping.stop:
        return _decision(
            context=context,
            policy=policy_config,
            diagnosis=diagnosis,
            decision_type=DecisionType.STOP,
            selected_action=None,
            reasons=[str(stopping.reason)],
            candidates=candidates,
            fallback_used=False,
            risk_checks_passed=False,
        )

    if stopping.forced_action:
        candidate = candidates[stopping.forced_action]
        blocking = [
            reason
            for reason in candidate.eligibility_reasons
            if not reason.startswith("insufficient_contextual_support")
        ]
        if blocking:
            return _decision(
                context=context,
                policy=policy_config,
                diagnosis=diagnosis,
                decision_type=DecisionType.BLOCK,
                selected_action=None,
                reasons=[str(stopping.reason), *blocking],
                candidates=candidates,
                fallback_used=True,
                risk_checks_passed=False,
            )
        return _decision(
            context=context,
            policy=policy_config,
            diagnosis=diagnosis,
            decision_type=DecisionType.FALLBACK,
            selected_action=stopping.forced_action,
            reasons=[str(stopping.reason), "manual_action_only"],
            candidates=candidates,
            fallback_used=True,
            risk_checks_passed=True,
        )

    eligible = [candidate for candidate in candidates.values() if candidate.eligible]
    minimum_ev = float(
        policy_config["decision"]["minimum_expected_value_inr"]
    )
    eligible = [
        candidate
        for candidate in eligible
        if candidate.expected_value_inr >= minimum_ev
    ]
    if not eligible:
        fallback = str(
            policy_config["decision"]["no_eligible_action_fallback"]
        )
        fallback_candidate = candidates[fallback]
        fallback_blockers = [
            reason
            for reason in fallback_candidate.eligibility_reasons
            if not reason.startswith("insufficient_contextual_support")
        ]
        if not fallback_blockers:
            return _decision(
                context=context,
                policy=policy_config,
                diagnosis=diagnosis,
                decision_type=DecisionType.FALLBACK,
                selected_action=fallback,
                reasons=[
                    "no_supported_positive_value_action",
                    "manual_escalation_fallback",
                ],
                candidates=candidates,
                fallback_used=True,
                risk_checks_passed=True,
            )
        return _decision(
            context=context,
            policy=policy_config,
            diagnosis=diagnosis,
            decision_type=DecisionType.STOP,
            selected_action=None,
            reasons=["no_eligible_positive_value_action", *fallback_blockers],
            candidates=candidates,
            fallback_used=False,
            risk_checks_passed=False,
        )

    ranked = sorted(
        eligible,
        key=lambda candidate: (
            -candidate.expected_value_inr,
            interventions.index(candidate.action),
        ),
    )
    best = ranked[0]
    if len(ranked) > 1:
        second = ranked[1]
        ev_margin = best.expected_value_inr - second.expected_value_inr
        probability_margin = best.probability - second.probability
        uncertain = (
            ev_margin
            < float(
                policy_config["decision"][
                    "minimum_expected_value_margin_inr"
                ]
            )
            or probability_margin
            < float(
                policy_config["decision"]["minimum_probability_margin"]
            )
        )
    else:
        ev_margin = None
        probability_margin = None
        uncertain = True

    if uncertain:
        preferred = str(
            policy_config["decision"]["preferred_fallback_action"]
        )
        preferred_candidate = candidates[preferred]
        selected = (
            preferred_candidate
            if preferred_candidate in eligible
            else best
        )
        return _decision(
            context=context,
            policy=policy_config,
            diagnosis=diagnosis,
            decision_type=DecisionType.FALLBACK,
            selected_action=selected.action,
            reasons=[
                "decision_uncertain",
                f"expected_value_margin:{ev_margin}",
                f"probability_margin:{probability_margin}",
            ],
            candidates=candidates,
            fallback_used=True,
            risk_checks_passed=True,
        )

    return _decision(
        context=context,
        policy=policy_config,
        diagnosis=diagnosis,
        decision_type=DecisionType.ALLOW,
        selected_action=best.action,
        reasons=[
            "eligible",
            "contextually_supported_or_manual",
            "highest_risk_adjusted_expected_value",
            f"expected_value_margin:{round(float(ev_margin), 2)}",
            f"probability_margin:{round(float(probability_margin), 8)}",
        ],
        candidates=candidates,
        fallback_used=False,
        risk_checks_passed=True,
    )
