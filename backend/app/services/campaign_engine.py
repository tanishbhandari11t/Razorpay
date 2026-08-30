from __future__ import annotations

"""Campaign engine scaffolding — Churnkey cadence + Novu-style delayed steps."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.services.automation_state import load_automation_state

ROOT = Path(__file__).resolve().parents[3]
CAMPAIGN_PATH = ROOT / "ml" / "config" / "recovery_campaigns.yaml"


@lru_cache(maxsize=1)
def load_campaigns() -> dict[str, Any]:
    return yaml.safe_load(CAMPAIGN_PATH.read_text(encoding="utf-8"))


def campaign_overview() -> dict[str, Any]:
    config = load_campaigns()
    automation = load_automation_state()
    campaigns = []
    for key, value in (config.get("campaigns") or {}).items():
        steps = value.get("steps") or []
        campaigns.append(
            {
                "id": key,
                "name": value.get("name", key),
                "enabled": bool(value.get("enabled", True))
                and bool(automation.get("campaigns_enabled", True)),
                "step_count": len(steps),
                "steps": [
                    {
                        "id": step.get("id"),
                        "at": step.get("at"),
                        "action": step.get("action"),
                        "communication": bool(step.get("communication")),
                    }
                    for step in steps
                ],
                "audience": value.get("audience") or {},
            }
        )
    return {
        "version": config.get("version"),
        "automation_enabled": bool(automation.get("recover_revenue_enabled")),
        "campaigns": campaigns,
        "defaults": config.get("defaults") or {},
        "note": (
            "Campaigns schedule diagnose → intervene → remind → escalate. "
            "Provider execution remains blocked while the system is in shadow mode."
        ),
    }


def select_campaign_for_case(
    *,
    amount_inr: float,
    fraud: bool = False,
    failure_class: str | None = None,
) -> dict[str, Any] | None:
    config = load_campaigns()
    if fraud:
        camp = (config.get("campaigns") or {}).get("fraud_block")
        if camp:
            return {"id": "fraud_block", **camp}
    if amount_inr >= 5000:
        camp = (config.get("campaigns") or {}).get("high_value_review")
        if camp:
            return {"id": "high_value_review", **camp}
    camp = (config.get("campaigns") or {}).get("payment_failed_default")
    if camp:
        return {"id": "payment_failed_default", **camp}
    return None
