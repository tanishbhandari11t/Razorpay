from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "real_outcome_schema.yaml"


@lru_cache(maxsize=1)
def load_real_outcome_schema() -> dict[str, Any]:
    schema = yaml.safe_load(SCHEMA_PATH.read_text(encoding="utf-8"))
    if schema["safety"]["controlled_execution_authorized"]:
        raise RuntimeError("Phase 14 outcome collection cannot authorize execution")
    if schema["safety"]["provider_actions_enabled"]:
        raise RuntimeError("Phase 14 outcome collection cannot enable providers")
    return schema


@dataclass(frozen=True)
class RecoveryOutcomeRecord:
    decision_id: str
    payment_id: str
    action: str | None
    decision_probability: float | None
    decision_margin: float | None
    policy_version: str
    model_version: str
    attempted: bool
    attempted_at: str | None
    failure_timestamp: str | None
    payment_status_after_24h: str | None
    payment_status_after_48h: str | None
    outcome_state: str
    outcome_at: str | None
    payment_recovered: bool | None
    recovered_amount_minor: int
    recovery_timestamp: str | None
    time_to_recovery_seconds: int | None
    observation_window_starts_at: str
    observation_window_ends_at: str
    outcome_source: str
    data_source: str
    natural_recovery_observed: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_outcome_record(record: RecoveryOutcomeRecord) -> None:
    schema = load_real_outcome_schema()
    if record.outcome_state not in schema["states"]:
        raise ValueError("Unknown recovery outcome state")
    if record.outcome_source not in schema["outcome_sources"]:
        raise ValueError("Unknown recovery outcome source")
    if record.data_source not in schema["data_sources"]:
        raise ValueError("Unknown recovery data source")
    if record.recovered_amount_minor < 0:
        raise ValueError("Recovered amount cannot be negative")
    if record.payment_recovered is True and record.outcome_state != "recovered":
        raise ValueError("Recovered labels require the recovered terminal state")
    if record.outcome_state == "recovered" and not record.attempted:
        raise ValueError(
            "Unattempted actions cannot receive intervention recovery labels"
        )
    if (
        record.outcome_state == "no_recovery_observed"
        and record.payment_recovered is False
    ):
        raise ValueError("Timeout cannot be labeled recovered=false")
    if record.outcome_state == "no_recovery_observed" and record.payment_recovered is True:
        raise ValueError("Timeout cannot be labeled recovered")


def training_eligibility(record: RecoveryOutcomeRecord) -> dict[str, Any]:
    validate_outcome_record(record)
    schema = load_real_outcome_schema()
    requirements = schema["training_eligibility"]
    reasons: list[str] = []
    if requirements["require_terminal_state"] and record.outcome_state not in schema[
        "terminal_states"
    ]:
        reasons.append("outcome_not_terminal")
    if requirements["require_attempted_action"] and not record.attempted:
        reasons.append("action_not_attempted")
    if record.data_source != requirements["require_data_source"]:
        reasons.append("data_source_not_real_controlled")
    if record.natural_recovery_observed and record.outcome_state != "recovered":
        reasons.append("independent_or_late_recovery_not_attributed")
    return {"eligible": not reasons, "reasons": reasons}


def retraining_gate(
    eligible_rows: int,
    *,
    coverage_passed: bool,
) -> dict[str, Any]:
    schema = load_real_outcome_schema()
    requirements = schema["training_eligibility"]
    experimental = requirements["experimental_challenger_rows"]
    if eligible_rows < int(experimental["minimum"]):
        stage = "no_retraining"
    elif eligible_rows <= int(experimental["maximum"]):
        stage = "experimental_challenger"
    else:
        stage = "production_candidate"
    return {
        "stage": stage,
        "eligible_rows": eligible_rows,
        "coverage_passed": coverage_passed,
        "authorized": False,
        "reason": (
            "Phase 14 collects evidence only; retraining stays blocked until "
            "a later challenger phase authorizes it."
        ),
    }
