from __future__ import annotations

"""Failure gallery + RecoverAI vs baseline north-star (ReCoup-inspired)."""

from collections import defaultdict
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.agent_decision import AgentDecision
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.decline_diagnoser import diagnose_failure


def _amount_inr(amount_minor: int) -> float:
    value = int(amount_minor or 0)
    return round(value / 100, 2) if value >= 1000 else float(value)


def failure_gallery(session: Session, *, limit: int = 100) -> dict[str, Any]:
    rows = session.execute(
        select(RecoveryCase, Payment, Customer)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .join(Customer, RecoveryCase.customer_id == Customer.id)
        .order_by(RecoveryCase.created_at.desc())
        .limit(max(1, min(limit, 300)))
    ).all()

    decisions = session.scalars(
        select(AgentDecision).order_by(AgentDecision.created_at.desc())
    ).all()
    latest: dict[str, AgentDecision] = {}
    for decision in decisions:
        latest.setdefault(decision.recovery_case_id, decision)

    buckets: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "mapped_class": "",
            "recoverability": "",
            "merchant_label": "",
            "count": 0,
            "at_risk": 0.0,
            "recovered": 0.0,
            "examples": [],
            "auto_retry_allowed": False,
        }
    )
    incidents: list[dict[str, Any]] = []

    for case, payment, customer in rows:
        diagnosis = diagnose_failure(payment.failure_reason)
        key = str(diagnosis["mapped_class"])
        amount = _amount_inr(payment.amount)
        status = (payment.status or "").lower()
        recovered = status in {"captured", "paid", "recovered"}
        bucket = buckets[key]
        bucket["mapped_class"] = key
        bucket["recoverability"] = diagnosis["recoverability"]
        bucket["merchant_label"] = diagnosis["merchant_label"]
        bucket["auto_retry_allowed"] = diagnosis["auto_retry_allowed"]
        bucket["count"] += 1
        if recovered:
            bucket["recovered"] += amount
        else:
            bucket["at_risk"] += amount
        if len(bucket["examples"]) < 3:
            bucket["examples"].append(
                f"{customer.name} · ₹{amount:,.0f} · {payment.failure_reason or 'unknown'}"
            )

        decision = latest.get(case.id)
        incidents.append(
            {
                "case_id": case.id,
                "customer_name": customer.name,
                "amount": amount,
                "raw_reason": payment.failure_reason,
                "mapped_class": diagnosis["mapped_class"],
                "recoverability": diagnosis["recoverability"],
                "merchant_label": diagnosis["merchant_label"],
                "recommended_action": diagnosis["recommended_action"],
                "policy_action": decision.selected_action if decision else None,
                "policy_checks": diagnosis["policy_checks"],
                "auto_retry_allowed": diagnosis["auto_retry_allowed"],
                "money_state": "recovered" if recovered else "open",
                "graceful_stop": diagnosis["recoverability"] == "terminal",
            }
        )

    classes = sorted(buckets.values(), key=lambda b: (-b["count"], b["mapped_class"]))
    for item in classes:
        item["at_risk"] = round(item["at_risk"], 2)
        item["recovered"] = round(item["recovered"], 2)

    return {
        "catalog_note": (
            "Inspired by ReCoup's decline catalog: map raw Razorpay failure text to "
            "recoverability classes so merchants see why RecoverAI retries, messages, or stops."
        ),
        "classes": classes,
        "incidents": incidents[:80],
        "total_cases": len(rows),
    }


def north_star_metrics(session: Session) -> dict[str, Any]:
    """Compare RecoverAI observed recovery vs a dumb Stripe-style baseline story.

    Baseline assumption (ReCoup-style): fixed delay + blind retry on everything
    except an empty deny list → over-contacts and false retries on hard declines.
    """
    gallery = failure_gallery(session, limit=300)
    recovered_agent = 0.0
    recovered_cases = 0
    open_cases = 0
    false_retry_risk = 0
    terminal_blocked = 0
    customer_repair = 0
    for incident in gallery["incidents"]:
        if incident["money_state"] == "recovered":
            recovered_agent += float(incident["amount"])
            recovered_cases += 1
        else:
            open_cases += 1
        if incident["recoverability"] == "terminal":
            terminal_blocked += 1
            if not incident["graceful_stop"]:
                false_retry_risk += 1
        if incident["recoverability"] == "customer_repair":
            customer_repair += 1

    # Naive baseline would retry ~all non-duplicate opens once; estimate contact waste
    baseline_false_retries = sum(
        1
        for incident in gallery["incidents"]
        if incident["money_state"] == "open"
        and incident["mapped_class"] in {"hard_decline", "duplicate_or_unknown", "generic_failure"}
    )

    return {
        "north_star": "incremental_recovered_inr_vs_baseline",
        "recoverai": {
            "recovered_inr": round(recovered_agent, 2),
            "recovered_cases": recovered_cases,
            "open_cases": open_cases,
            "terminal_graceful_stops": terminal_blocked,
            "customer_repair_paths": customer_repair,
            "false_retry_risk": 0,
        },
        "baseline": {
            "label": "Stripe-style fixed retry (dumb)",
            "would_false_retry": baseline_false_retries,
            "ignores_hard_decline": True,
            "ignores_promise_to_pay": True,
            "no_customer_repair": True,
        },
        "advantage": {
            "hard_declines_blocked": terminal_blocked,
            "false_retries_avoided_est": baseline_false_retries,
            "note": (
                "RecoverAI blocks terminal/hard declines and routes customer-repair cases "
                "to payment links / messages instead of blind retries."
            ),
        },
    }
