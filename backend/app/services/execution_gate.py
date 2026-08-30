from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from app.config.settings import get_settings


REPO_ROOT = Path(__file__).resolve().parents[3]
EXECUTION_GATE_PATH = REPO_ROOT / "ml" / "config" / "execution_gate.yaml"


class ExecutionGateError(RuntimeError):
    pass


@dataclass(frozen=True)
class ExecutionGateDecision:
    allowed: bool
    mode: str
    action: str | None
    reason: str


@lru_cache(maxsize=1)
def load_execution_gate() -> dict[str, Any]:
    config = yaml.safe_load(EXECUTION_GATE_PATH.read_text(encoding="utf-8"))
    configured_mode = str(config["execution"]["mode"])
    environment_mode = get_settings().execution_mode
    if configured_mode != environment_mode:
        raise ExecutionGateError(
            "Execution mode mismatch between environment and frozen config"
        )
    if (
        configured_mode == "controlled"
        and not config["execution"]["controlled_execution_authorized"]
    ):
        raise ExecutionGateError(
            "Controlled execution is not authorized by the frozen gate"
        )
    return config


def check_execution_gate(action: str | None) -> ExecutionGateDecision:
    config = load_execution_gate()
    mode = str(config["execution"]["mode"])
    if mode == "shadow":
        return ExecutionGateDecision(
            allowed=False,
            mode=mode,
            action=action,
            reason="execution_mode_shadow",
        )
    if mode == "dry_run":
        return ExecutionGateDecision(
            allowed=False,
            mode=mode,
            action=action,
            reason="execution_mode_dry_run",
        )
    if not config["execution"]["provider_actions_enabled"]:
        return ExecutionGateDecision(
            allowed=False,
            mode=mode,
            action=action,
            reason="provider_actions_disabled",
        )
    return ExecutionGateDecision(
        allowed=True,
        mode=mode,
        action=action,
        reason="controlled_execution_authorized",
    )
