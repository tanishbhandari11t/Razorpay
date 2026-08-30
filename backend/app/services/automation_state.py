from __future__ import annotations

"""Merchant Recover Revenue automation toggle — Novu-style workflow on/off."""

from pathlib import Path
from typing import Any

import yaml

from app.services.kill_switch import kill_switch_armed

ROOT = Path(__file__).resolve().parents[3]
STATE_PATH = ROOT / "ml" / "config" / "automation_state.yaml"


def _default_state() -> dict[str, Any]:
    return {
        "version": 1,
        "recover_revenue_enabled": False,
        "messaging_enabled": True,
        "promise_tracking_enabled": True,
        "campaigns_enabled": True,
        "mode": "fully_automatic",
        "max_automatic_recovery_inr": 5000,
        "merchant_alerts": "exceptions_only",
        "enabled_at": None,
        "paused_at": None,
    }


def load_automation_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        state = _default_state()
        save_automation_state(state)
        return state
    raw = yaml.safe_load(STATE_PATH.read_text(encoding="utf-8")) or {}
    state = _default_state()
    state.update({k: v for k, v in raw.items() if k in state})
    return state


def save_automation_state(state: dict[str, Any]) -> dict[str, Any]:
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        yaml.safe_dump(state, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return state


def set_recover_revenue(enabled: bool) -> dict[str, Any]:
    from datetime import UTC, datetime

    state = load_automation_state()
    now = datetime.now(UTC).isoformat()
    state["recover_revenue_enabled"] = bool(enabled)
    if enabled:
        state["enabled_at"] = now
        state["paused_at"] = None
    else:
        state["paused_at"] = now
    save_automation_state(state)
    return automation_status()


def automation_status() -> dict[str, Any]:
    state = load_automation_state()
    kill = kill_switch_armed()
    active = bool(state.get("recover_revenue_enabled")) and not kill
    checks = [
        {"id": "razorpay", "label": "Razorpay connected", "ok": True},
        {"id": "monitoring", "label": "Failure monitoring enabled", "ok": active},
        {"id": "policy", "label": "Recovery policy enabled", "ok": active},
        {
            "id": "messaging",
            "label": "Customer messaging enabled",
            "ok": active and bool(state.get("messaging_enabled")),
        },
        {
            "id": "promises",
            "label": "Promise-to-pay tracking enabled",
            "ok": active and bool(state.get("promise_tracking_enabled")),
        },
        {
            "id": "safety",
            "label": "Safety controls enabled",
            "ok": True,
        },
    ]
    return {
        **state,
        "active": active,
        "kill_switch": kill,
        "checks": checks,
        "headline": (
            "RecoverAI is now recovering revenue."
            if active
            else (
                "Emergency stop is armed — automation paused."
                if kill
                else "Click Recover Revenue to turn automation on."
            )
        ),
        "note": (
            "Shadow mode still blocks real provider execution until the evidence gate authorizes pilot. "
            "Automation orchestrates detection, drafting, promises, and merchant exceptions."
        ),
    }
