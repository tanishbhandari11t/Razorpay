from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from ml.src.record_recovery_outcome import load_real_outcome_schema

REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_CONFIG_PATH = (
    REPO_ROOT / "ml" / "config" / "outcome_observation.yaml"
)


@lru_cache(maxsize=1)
def load_outcome_observation_config() -> dict[str, Any]:
    config = yaml.safe_load(OBSERVATION_CONFIG_PATH.read_text(encoding="utf-8"))
    safety = config["safety"]
    if safety["controlled_execution_authorized"]:
        raise RuntimeError("Outcome observation cannot authorize execution")
    if safety["provider_actions_enabled"]:
        raise RuntimeError("Outcome observation cannot enable providers")
    if safety["phase15_authorized"]:
        raise RuntimeError("Phase 14 cannot authorize Phase 15")
    if safety["timeout_sets_payment_recovered_false"]:
        raise RuntimeError("Timeout must not be labeled recovered=false")
    return config


@dataclass(frozen=True)
class OutcomeTransition:
    previous_state: str
    next_state: str
    observed_at: datetime
    source: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "previous_state": self.previous_state,
            "next_state": self.next_state,
            "observed_at": self.observed_at.isoformat(),
            "source": self.source,
            "reason": self.reason,
        }


def transition_outcome(
    current_state: str,
    next_state: str,
    *,
    observed_at: datetime,
    source: str,
    reason: str,
) -> OutcomeTransition:
    schema = load_real_outcome_schema()
    if source not in schema["outcome_sources"]:
        raise ValueError("Unknown outcome observation source")
    if current_state in schema["terminal_states"]:
        raise ValueError("Terminal recovery outcomes are immutable")
    allowed = schema["transitions"].get(current_state, [])
    if next_state not in allowed:
        raise ValueError(
            f"Invalid outcome transition: {current_state} -> {next_state}"
        )
    return OutcomeTransition(
        previous_state=current_state,
        next_state=next_state,
        observed_at=observed_at,
        source=source,
        reason=reason,
    )


def time_to_recovery_seconds(
    *,
    failure_timestamp: datetime | None,
    recovery_timestamp: datetime | None,
) -> int | None:
    if failure_timestamp is None or recovery_timestamp is None:
        return None
    elapsed = (recovery_timestamp - failure_timestamp).total_seconds()
    return int(elapsed) if elapsed >= 0 else None


def status_checkpoint_due(
    *,
    anchor_at: datetime | None,
    observed_at: datetime,
    hours: int,
    existing_status: str | None,
) -> bool:
    if existing_status is not None or anchor_at is None:
        return False
    return observed_at >= anchor_at + timedelta(hours=hours)


def checkpoint_status_updates(
    *,
    anchor_at: datetime | None,
    observed_at: datetime,
    payment_status: str,
    status_after_24h: str | None,
    status_after_48h: str | None,
) -> dict[str, str]:
    schema = load_real_outcome_schema()
    existing = {
        24: status_after_24h,
        48: status_after_48h,
    }
    updates: dict[str, str] = {}
    for hours in schema["observation"]["checkpoint_hours"]:
        hours = int(hours)
        if status_checkpoint_due(
            anchor_at=anchor_at,
            observed_at=observed_at,
            hours=hours,
            existing_status=existing.get(hours),
        ):
            updates[f"payment_status_after_{hours}h"] = payment_status
    return updates


def recovered_status(payment_status: str) -> bool:
    config = load_outcome_observation_config()
    return payment_status.lower() in {
        status.lower() for status in config["recovered_statuses"]
    }


def capture_attribution(
    *,
    payment_status: str,
    attempted: bool,
    attempted_at: datetime | None,
    observed_at: datetime,
    window_starts_at: datetime,
    window_ends_at: datetime,
    outcome_state: str,
) -> str:
    if not recovered_status(payment_status):
        return "none"
    if outcome_state in load_real_outcome_schema()["terminal_states"]:
        return "observational"
    if (
        not attempted
        or attempted_at is None
        or observed_at < attempted_at
        or observed_at > window_ends_at
        or attempted_at < window_starts_at
        or outcome_state != "waiting_for_outcome"
    ):
        return "observational"
    return "attributed"


def outcome_label_kind(
    *,
    outcome_state: str,
    attempted: bool,
    payment_recovered: bool | None,
    natural_recovery_observed: bool,
    data_source: str,
) -> str:
    if outcome_state == "recovered" and attempted and payment_recovered is True:
        return "attributed_intervention_recovery"
    if natural_recovery_observed and payment_recovered is not True:
        return "observational_recovery"
    if outcome_state == "no_recovery_observed":
        return "no_recovery_observed"
    if outcome_state == "unknown":
        return "unknown"
    if data_source == "synthetic":
        return "pending"
    return "pending"


def observed_terminal_state(
    *,
    payment_status: str,
    attempted: bool,
    observed_at: datetime,
    observation_window_ends_at: datetime,
) -> str | None:
    if (
        attempted
        and recovered_status(payment_status)
        and observed_at <= observation_window_ends_at
    ):
        return "recovered"
    if observed_at >= observation_window_ends_at:
        return "no_recovery_observed"
    return None
