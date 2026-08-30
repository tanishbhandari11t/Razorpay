from __future__ import annotations

from collections import Counter
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.recovery_case import RecoveryCase
from app.models.webhook_event import WebhookEvent


PRIMARY_POLICY = "recovery_policy_v3"
CHALLENGER_POLICY = "recovery_policy_v3_v2_online"


def compare_case_decisions(
    session: Session,
    recovery_case_id: str,
) -> dict[str, Any]:
    decisions = session.scalars(
        select(AgentDecision).where(
            AgentDecision.recovery_case_id == recovery_case_id,
            AgentDecision.execution_mode == "shadow",
            AgentDecision.policy_version.in_(
                {PRIMARY_POLICY, CHALLENGER_POLICY}
            ),
        )
    ).all()
    by_policy = {decision.policy_version: decision for decision in decisions}
    primary = by_policy.get(PRIMARY_POLICY)
    challenger = by_policy.get(CHALLENGER_POLICY)

    def serialize(decision: AgentDecision | None) -> dict[str, Any] | None:
        if decision is None:
            return None
        return {
            "decision_id": decision.id,
            "model_version": decision.model_version,
            "policy_version": decision.policy_version,
            "selected_action": decision.selected_action or "no_action",
            "decision_margin": decision.decision_margin,
            "decision_margin_type": (
                "expected_value_inr"
                if decision.policy_version == CHALLENGER_POLICY
                else "probability"
            ),
            "fallback_used": decision.fallback_used,
            "failure_class": decision.failure_class,
            "risk_checks_passed": decision.risk_checks_passed,
            "executed": False,
        }

    primary_payload = serialize(primary)
    challenger_payload = serialize(challenger)
    paired = primary is not None and challenger is not None
    return {
        "recovery_case_id": recovery_case_id,
        "primary": primary_payload,
        "challenger": challenger_payload,
        "paired": paired,
        "action_agreement": (
            (primary.selected_action or "no_action")
            == (challenger.selected_action or "no_action")
            if paired
            else None
        ),
        "margin_delta_challenger_minus_primary": (
            None
        ),
        "margin_delta_reason": (
            "Margins use different units and are not subtracted"
            if paired
            else None
        ),
    }


def shadow_ab_metrics(session: Session) -> dict[str, Any]:
    rows = session.execute(
        select(AgentDecision, RecoveryCase)
        .join(
            RecoveryCase,
            RecoveryCase.id == AgentDecision.recovery_case_id,
        )
        .join(
            WebhookEvent,
            WebhookEvent.razorpay_event_id
            == RecoveryCase.source_event_id,
        )
        .where(
            AgentDecision.execution_mode == "shadow",
            AgentDecision.policy_version.in_(
                {PRIMARY_POLICY, CHALLENGER_POLICY}
            ),
            WebhookEvent.razorpay_event_id.not_like("evt_postgres_%"),
        )
    ).all()
    by_case: dict[str, dict[str, AgentDecision]] = {}
    for decision, recovery_case in rows:
        by_case.setdefault(recovery_case.id, {})[
            decision.policy_version
        ] = decision
    pairs = [
        values
        for values in by_case.values()
        if PRIMARY_POLICY in values and CHALLENGER_POLICY in values
    ]
    agreements = [
        (pair[PRIMARY_POLICY].selected_action or "no_action")
        == (pair[CHALLENGER_POLICY].selected_action or "no_action")
        for pair in pairs
    ]
    return {
        "primary_policy": PRIMARY_POLICY,
        "challenger_policy": CHALLENGER_POLICY,
        "primary_cases": sum(
            PRIMARY_POLICY in values for values in by_case.values()
        ),
        "challenger_cases": sum(
            CHALLENGER_POLICY in values for values in by_case.values()
        ),
        "paired_cases": len(pairs),
        "action_agreement_rate": (
            sum(agreements) / len(agreements) if agreements else None
        ),
        "primary_action_distribution": dict(
            Counter(
                pair[PRIMARY_POLICY].selected_action or "no_action"
                for pair in pairs
            )
        ),
        "challenger_action_distribution": dict(
            Counter(
                pair[CHALLENGER_POLICY].selected_action or "no_action"
                for pair in pairs
            )
        ),
        "provider_calls": 0,
        "execution_mode": "shadow",
    }
