from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = (
    REPO_ROOT / "ml" / "config" / "intervention_policy.yaml"
)


class FailureClass(StrEnum):
    FRAUD_RISK = "fraud_risk"
    TECHNICAL_RETRYABLE = "technical_retryable"
    INSUFFICIENT_FUNDS = "insufficient_funds"
    AUTHENTICATION_REQUIRED = "authentication_required"
    MERCHANT_CONFIGURATION = "merchant_configuration"
    PAYMENT_METHOD_RESTRICTED = "payment_method_restricted"
    CUSTOMER_DECLINE = "customer_decline"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class FailureDiagnosis:
    failure_class: FailureClass
    normalized_reason: str
    matched_rule: str
    retryable: bool


def load_intervention_policy(
    path: Path = DEFAULT_POLICY_PATH,
) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def normalize_failure_reason(reason: str | None) -> str:
    normalized = re.sub(r"[^a-z0-9]+", "_", str(reason or "").strip().lower())
    return normalized.strip("_") or "unknown"


def classify_failure(
    reason: str | None,
    *,
    fraud_flag: int = 0,
    config: dict[str, Any] | None = None,
) -> FailureDiagnosis:
    policy = config or load_intervention_policy()
    normalized = normalize_failure_reason(reason)
    if int(fraud_flag) == 1:
        return FailureDiagnosis(
            failure_class=FailureClass.FRAUD_RISK,
            normalized_reason=normalized,
            matched_rule="fraud_flag",
            retryable=False,
        )

    readable = normalized.replace("_", " ")
    for class_name, rules in policy["failure_taxonomy"].items():
        if class_name == FailureClass.UNKNOWN:
            continue
        exact = {
            normalize_failure_reason(value)
            for value in rules.get("exact", [])
        }
        if normalized in exact:
            failure_class = FailureClass(class_name)
            return FailureDiagnosis(
                failure_class=failure_class,
                normalized_reason=normalized,
                matched_rule=f"exact:{normalized}",
                retryable=(
                    failure_class == FailureClass.TECHNICAL_RETRYABLE
                ),
            )
        for fragment in rules.get("contains", []):
            if str(fragment).lower() in readable:
                failure_class = FailureClass(class_name)
                return FailureDiagnosis(
                    failure_class=failure_class,
                    normalized_reason=normalized,
                    matched_rule=f"contains:{fragment}",
                    retryable=(
                        failure_class
                        == FailureClass.TECHNICAL_RETRYABLE
                    ),
                )

    return FailureDiagnosis(
        failure_class=FailureClass.UNKNOWN,
        normalized_reason=normalized,
        matched_rule="default:unknown",
        retryable=False,
    )
