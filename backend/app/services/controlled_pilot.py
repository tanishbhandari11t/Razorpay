from __future__ import annotations

"""
Phase 17 controlled execution pilot helpers.

Everything remains blocked while execution.mode=shadow and
controlled_execution_authorized=false. The kill switch forces BLOCK.

Pilot knobs live in ml/config/controlled_pilot.yaml so the frozen
execution_gate.yaml Phase 11 baseline hash stays intact.
"""

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.services.execution_gate import (
    ExecutionGateDecision,
    ExecutionGateError,
    load_execution_gate,
)
from app.services.kill_switch import kill_switch_armed


REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "controlled_pilot.yaml"


@dataclass(frozen=True)
class ControlledPilotDecision:
    allowed: bool
    action: str | None
    reason: str
    checks: dict[str, bool]


@lru_cache(maxsize=1)
def load_controlled_pilot_config() -> dict[str, Any]:
    config = yaml.safe_load(PILOT_CONFIG_PATH.read_text(encoding="utf-8"))
    if config["execution"]["controlled_execution_authorized"]:
        raise ExecutionGateError(
            "Controlled pilot config cannot authorize execution"
        )
    if config["controlled_pilot"]["enabled"] or config.get("pilot", {}).get(
        "enabled"
    ):
        raise ExecutionGateError("Controlled pilot must stay disabled")
    if config["safety"]["phase17_authorized"]:
        raise ExecutionGateError("Phase 17 remains unauthorized")
    return config


def evaluate_controlled_pilot(
    *,
    action: str | None,
    model_ready: bool,
    taxonomy_known: bool,
    feature_support_ok: bool,
    fraud_pass: bool,
    risk_pass: bool,
    decision_margin_ok: bool,
    amount_inr: float,
    attempt_count: int,
    cooldown_satisfied: bool,
    daily_actions: int,
    customer_actions: int,
) -> ControlledPilotDecision:
    gate = load_execution_gate()
    pilot_config = load_controlled_pilot_config()
    execution = {
        **gate["execution"],
        **pilot_config["execution"],
    }
    # Frozen gate wins on mode and authorization.
    execution["mode"] = gate["execution"]["mode"]
    execution["controlled_execution_authorized"] = gate["execution"][
        "controlled_execution_authorized"
    ]
    execution["provider_actions_enabled"] = gate["execution"][
        "provider_actions_enabled"
    ]
    pilot = pilot_config["controlled_pilot"]
    allowlisted = set(
        pilot_config.get("allowlist", {}).get("actions")
        or [pilot["initial_action"]]
    )
    kill_switch_is_armed = kill_switch_armed()
    checks = {
        "global_kill_switch_off": not kill_switch_is_armed,
        "mode_controlled": str(execution["mode"]) == "controlled",
        "controlled_authorized": bool(
            execution["controlled_execution_authorized"]
        ),
        "provider_actions_enabled": bool(execution["provider_actions_enabled"]),
        "pilot_enabled": bool(pilot["enabled"]),
        "model_ready": model_ready,
        "taxonomy_known": taxonomy_known,
        "feature_support": feature_support_ok,
        "fraud_check": fraud_pass,
        "risk_check": risk_pass,
        "decision_margin": decision_margin_ok,
        "attempt_budget": attempt_count < int(pilot["max_attempts"]),
        "cooldown": cooldown_satisfied,
        "amount_limit": amount_inr <= float(pilot["max_amount_inr"]),
        "daily_limit": daily_actions < int(pilot["daily_action_limit"]),
        "customer_limit": customer_actions < int(pilot["customer_action_limit"]),
        "action_allowlisted": action in allowlisted,
    }
    if not checks["global_kill_switch_off"]:
        return ControlledPilotDecision(
            allowed=False,
            action=action,
            reason="global_kill_switch",
            checks=checks,
        )
    if str(execution["mode"]) == "disabled":
        return ControlledPilotDecision(
            allowed=False,
            action=action,
            reason="execution_disabled",
            checks=checks,
        )
    if not all(checks.values()):
        failed = [name for name, passed in checks.items() if not passed]
        return ControlledPilotDecision(
            allowed=False,
            action=action,
            reason=f"blocked:{','.join(failed)}",
            checks=checks,
        )
    return ControlledPilotDecision(
        allowed=True,
        action=action,
        reason="controlled_pilot_allowed",
        checks=checks,
    )


def assert_pilot_still_blocked() -> ExecutionGateDecision:
    """Hard safety assertion used by tests and Phase 17 scaffolding."""
    gate = load_execution_gate()
    pilot = load_controlled_pilot_config()
    if gate["execution"]["mode"] != "shadow":
        raise ExecutionGateError("Phase 17 scaffold expects shadow mode")
    if gate["execution"]["controlled_execution_authorized"]:
        raise ExecutionGateError("Controlled execution must stay unauthorized")
    if pilot["controlled_pilot"]["enabled"]:
        raise ExecutionGateError("Controlled pilot must stay disabled")
    return ExecutionGateDecision(
        allowed=False,
        mode="shadow",
        action=None,
        reason="execution_mode_shadow",
    )
