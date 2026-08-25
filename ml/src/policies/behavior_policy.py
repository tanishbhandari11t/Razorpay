from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "logging_policy.yaml"


@dataclass(frozen=True)
class PolicyDecision:
    chosen_intervention: str
    base_policy_intervention: str
    policy_probability: float
    policy_type: str


def load_policy_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    exploration_rate = float(config["exploration_rate"])
    if not 0 <= exploration_rate < 1:
        raise ValueError("Exploration rate must be in [0, 1)")
    interventions = config.get("eligible_interventions", [])
    if len(interventions) < 2 or len(interventions) != len(set(interventions)):
        raise ValueError("Eligible interventions must be unique")
    fractions = config["temporal_split"]
    if not math.isclose(
        sum(
            float(fractions[name])
            for name in ("train_fraction", "validation_fraction", "test_fraction")
        ),
        1.0,
        abs_tol=1e-9,
    ):
        raise ValueError("Temporal split fractions must sum to 1")
    return config


def _unit_interval(seed: int, *parts: object) -> float:
    material = "|".join([str(seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    integer = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return (integer + 0.5) / 2**64


def base_policy_action(
    features: Mapping[str, Any],
    config: dict[str, Any],
) -> str:
    if int(features["fraud_flag"]) == 1:
        return str(config["fraud_action"])

    rules = config["rules"]
    retry = rules["recent_reliable_retry"]
    if (
        int(features["has_previous_success"]) == 1
        and float(features["historical_success_rate"])
        >= float(retry["minimum_historical_success_rate"])
        and float(features["days_since_previous_success"])
        <= float(retry["maximum_days_since_previous_success"])
    ):
        return "retry_payment"

    payment_link = rules["strong_customer_payment_link"]
    if (
        int(features["has_prior_history"]) == 1
        and float(features["historical_success_rate"])
        >= float(payment_link["minimum_historical_success_rate"])
        and float(features["amount_vs_previous_avg"])
        > float(payment_link["minimum_amount_vs_previous_avg"])
    ):
        return "payment_link"

    whatsapp = rules["active_customer_whatsapp"]
    if int(features["transactions_last_30d"]) >= int(
        whatsapp["minimum_transactions_last_30d"]
    ):
        return "whatsapp_reminder"

    return str(rules["fallback_action"])


def choose_action(
    features: Mapping[str, Any],
    config: dict[str, Any],
) -> PolicyDecision:
    transaction_id = str(features["transaction_id"])
    base_action = base_policy_action(features, config)
    if base_action == config["fraud_action"]:
        return PolicyDecision(
            chosen_intervention=base_action,
            base_policy_intervention=base_action,
            policy_probability=1.0,
            policy_type="policy_blocked",
        )

    interventions = [str(value) for value in config["eligible_interventions"]]
    if base_action not in interventions:
        raise ValueError(f"Base policy chose ineligible action: {base_action}")
    alternatives = [
        intervention
        for intervention in interventions
        if intervention != base_action
    ]
    exploration_rate = float(config["exploration_rate"])
    branch_draw = _unit_interval(config["seed"], "exploration", transaction_id)

    if branch_draw < exploration_rate:
        action_draw = _unit_interval(
            config["seed"],
            "exploration-action",
            transaction_id,
        )
        index = min(int(action_draw * len(alternatives)), len(alternatives) - 1)
        return PolicyDecision(
            chosen_intervention=alternatives[index],
            base_policy_intervention=base_action,
            policy_probability=exploration_rate / len(alternatives),
            policy_type="exploration",
        )

    return PolicyDecision(
        chosen_intervention=base_action,
        base_policy_intervention=base_action,
        policy_probability=1 - exploration_rate,
        policy_type="behavior",
    )
