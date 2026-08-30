from __future__ import annotations

"""ReCoup-inspired decline diagnoser: pattern → recoverability → recommended action."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[3]
CATALOG_PATH = ROOT / "ml" / "config" / "decline_recoverability.yaml"


@lru_cache(maxsize=1)
def load_decline_catalog() -> dict[str, Any]:
    return yaml.safe_load(CATALOG_PATH.read_text(encoding="utf-8"))


def diagnose_failure(reason: str | None) -> dict[str, Any]:
    catalog = load_decline_catalog()
    defaults = catalog.get("defaults") or {}
    raw = (reason or "").strip().lower().replace("_", " ")
    matched = None
    confidence = "unknown"
    for entry in catalog.get("entries") or []:
        patterns = [str(p).lower() for p in (entry.get("patterns") or [])]
        for pattern in patterns:
            if pattern and pattern in raw:
                matched = entry
                confidence = "exact" if raw == pattern else "fuzzy"
                break
        if matched:
            break

    if matched is None:
        return {
            "raw_reason": reason,
            "mapped_class": defaults.get("mapped_class", "unknown"),
            "recoverability": defaults.get("recoverability", "unknown"),
            "recommended_action": defaults.get("recommended_action", "escalate_to_merchant"),
            "merchant_label": defaults.get("merchant_label", "Unknown failure"),
            "confidence": "unknown",
            "catalog_version": catalog.get("catalog_version"),
            "auto_retry_allowed": False,
            "policy_checks": ["unknown_no_silent_retry"],
        }

    recoverability = str(matched.get("recoverability") or "unknown")
    mapped = str(matched.get("mapped_class") or "unknown")
    rules = catalog.get("policy_rules") or {}
    never = set(rules.get("never_retry") or [])
    auto_retry = recoverability in {"silent_retry", "wait_then_retry", "wait_window"} and mapped not in never
    checks = ["catalog_matched"]
    if mapped in never or recoverability == "terminal":
        checks.append("hard_decline_blocked" if mapped == "hard_decline" else "terminal_no_retry")
        auto_retry = False
    elif recoverability == "unknown":
        checks.append("unknown_requires_review")
        auto_retry = False
    elif recoverability == "customer_repair":
        checks.append("customer_repair_preferred")
    elif auto_retry:
        checks.append("silent_retry_eligible")

    return {
        "raw_reason": reason,
        "mapped_class": mapped,
        "recoverability": recoverability,
        "recommended_action": matched.get("recommended_action"),
        "merchant_label": matched.get("merchant_label"),
        "confidence": confidence,
        "catalog_version": catalog.get("catalog_version"),
        "auto_retry_allowed": auto_retry,
        "policy_checks": checks,
    }
