from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TAXONOMY_PATH = (
    REPO_ROOT / "ml" / "config" / "razorpay_failure_taxonomy.yaml"
)


@dataclass(frozen=True)
class RazorpayFailureDiagnosis:
    raw_reason: str
    normalized_reason: str
    taxonomy: str
    subtype: str
    state: str
    confidence: str
    matched_rule: str
    retryable: bool
    safe_automation: str
    execution_allowed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_razorpay_reason(reason: str | None) -> str:
    normalized = re.sub(
        r"[^a-z0-9]+",
        "_",
        str(reason or "").strip().lower(),
    )
    return normalized.strip("_") or "unknown"


@lru_cache(maxsize=1)
def load_razorpay_taxonomy(
    path: Path = DEFAULT_TAXONOMY_PATH,
) -> dict[str, Any]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if config.get("default_taxonomy") != "unknown":
        raise RuntimeError("Razorpay taxonomy must fail closed to unknown")
    if config.get("unknown_execution_allowed") is not False:
        raise RuntimeError("Unknown Razorpay failures must block execution")
    return config


def classify_razorpay_failure(
    reason: str | None,
    *,
    fraud_flag: int = 0,
    config: dict[str, Any] | None = None,
) -> RazorpayFailureDiagnosis:
    taxonomy = config or load_razorpay_taxonomy()
    raw_reason = str(reason or "")
    normalized = normalize_razorpay_reason(reason)
    if int(fraud_flag) == 1:
        return RazorpayFailureDiagnosis(
            raw_reason=raw_reason,
            normalized_reason=normalized,
            taxonomy="fraud_risk",
            subtype="fraud_flag",
            state="KNOWN",
            confidence="deterministic",
            matched_rule="fraud_flag",
            retryable=False,
            safe_automation="never",
            execution_allowed=False,
        )

    mapping = taxonomy["mappings"].get(normalized)
    if mapping is None:
        explicit = taxonomy.get("explicit_unknowns", {}).get(normalized)
        return RazorpayFailureDiagnosis(
            raw_reason=raw_reason,
            normalized_reason=normalized,
            taxonomy="unknown",
            subtype="unknown",
            state="UNKNOWN",
            confidence="unknown",
            matched_rule=(
                f"explicit_unknown:{normalized}"
                if explicit is not None
                else "default:unknown"
            ),
            retryable=False,
            safe_automation="never",
            execution_allowed=False,
        )

    safe_automation = str(mapping["safe_automation"])
    return RazorpayFailureDiagnosis(
        raw_reason=raw_reason,
        normalized_reason=normalized,
        taxonomy=str(mapping["taxonomy"]),
        subtype=str(mapping["subtype"]),
        state="KNOWN",
        confidence="deterministic",
        matched_rule=f"exact:{normalized}",
        retryable=bool(mapping["retryable"]),
        safe_automation=safe_automation,
        execution_allowed=safe_automation == "conditional",
    )


def legacy_failure_class(diagnosis: RazorpayFailureDiagnosis) -> str:
    if diagnosis.taxonomy == "fraud_risk":
        return "fraud_risk"
    if diagnosis.taxonomy == "network":
        return "technical_retryable"
    if diagnosis.subtype == "insufficient_funds":
        return "insufficient_funds"
    if diagnosis.taxonomy == "authentication":
        return "authentication_required"
    if diagnosis.taxonomy == "merchant_configuration":
        return "merchant_configuration"
    if diagnosis.taxonomy == "payment_method":
        return "payment_method_restricted"
    if diagnosis.taxonomy in {"customer", "customer_payment_issue"}:
        return "customer_decline"
    if diagnosis.taxonomy == "bank_decline":
        return "bank_decline"
    return "unknown"
