from __future__ import annotations

from dataclasses import dataclass

from ml.src.policies.recovery_policy import DecisionType, RecoveryDecision


@dataclass(frozen=True)
class DecisionMarginResult:
    selected_action: str | None
    best_action_value_inr: float | None
    fallback_value_inr: float | None
    decision_margin_inr: float | None
    threshold_inr: float
    fallback_triggered: bool
    reason: str


def apply_decision_margin(
    decision: RecoveryDecision,
    *,
    fallback_action: str,
    threshold_inr: float,
) -> DecisionMarginResult:
    if threshold_inr < 0:
        raise ValueError("Decision-margin threshold must be non-negative")
    selected = decision.selected_action
    if decision.decision_type != DecisionType.ALLOW or selected is None:
        return DecisionMarginResult(
            selected_action=selected,
            best_action_value_inr=None,
            fallback_value_inr=None,
            decision_margin_inr=None,
            threshold_inr=float(threshold_inr),
            fallback_triggered=False,
            reason="margin_not_applicable_to_non_allow_decision",
        )
    if fallback_action not in decision.candidates:
        raise ValueError("Fallback action is missing from policy candidates")
    best_value = float(decision.expected_values[selected])
    fallback_value = float(decision.expected_values[fallback_action])
    margin = best_value - fallback_value
    fallback_candidate = decision.candidates[fallback_action]
    can_fallback = fallback_candidate.eligible and fallback_candidate.supported
    triggered = (
        selected != fallback_action
        and can_fallback
        and margin < float(threshold_inr)
    )
    return DecisionMarginResult(
        selected_action=fallback_action if triggered else selected,
        best_action_value_inr=best_value,
        fallback_value_inr=fallback_value,
        decision_margin_inr=margin,
        threshold_inr=float(threshold_inr),
        fallback_triggered=triggered,
        reason=(
            "decision_margin_below_threshold"
            if triggered
            else (
                "fallback_ineligible"
                if selected != fallback_action and not can_fallback
                else "decision_margin_passed"
            )
        ),
    )
