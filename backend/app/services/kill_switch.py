from __future__ import annotations

"""Runtime kill switch — Redis-backed, does not mutate frozen YAML gates."""

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from redis import Redis

from app.config.settings import get_settings


KILL_SWITCH_KEY = "recoverai:kill_switch"
REPO_ROOT = Path(__file__).resolve().parents[3]
PILOT_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "controlled_pilot.yaml"


@lru_cache(maxsize=1)
def _redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def _config_default_armed() -> bool:
    config = yaml.safe_load(PILOT_CONFIG_PATH.read_text(encoding="utf-8"))
    return bool(
        config.get("execution", {}).get("global_kill_switch")
        or config.get("kill_switch", {}).get("default_state")
    )


def kill_switch_armed() -> bool:
    config_armed = _config_default_armed()
    try:
        runtime = _redis().get(KILL_SWITCH_KEY)
    except Exception:
        runtime = None
    if runtime is None:
        return config_armed
    return str(runtime).strip().lower() in {"1", "true", "armed", "on"}


def set_kill_switch(armed: bool) -> dict[str, Any]:
    try:
        _redis().set(KILL_SWITCH_KEY, "1" if armed else "0")
        persisted = True
    except Exception:
        persisted = False
    return {
        "armed": kill_switch_armed() if persisted else armed,
        "persisted": persisted,
        "source": "redis" if persisted else "memory_fallback",
        "blocks_provider_calls": True,
    }


def kill_switch_status() -> dict[str, Any]:
    return {
        "armed": kill_switch_armed(),
        "blocks_provider_calls": True,
        "execution_mode": "shadow",
        "note": "Armed kill switch forces BLOCK on all provider actions.",
    }
