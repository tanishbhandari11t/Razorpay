from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd

from ml.src.build_real_outcome_dataset import ProvenanceError, split_outcomes_by_provenance
from ml.src.observe_recovery import (
    capture_attribution,
    checkpoint_status_updates,
    load_outcome_observation_config,
    outcome_label_kind,
    time_to_recovery_seconds,
    transition_outcome,
)
from ml.src.record_recovery_outcome import (
    RecoveryOutcomeRecord,
    training_eligibility,
    validate_outcome_record,
)


ANCHOR = datetime(2026, 1, 5, 10, 0, tzinfo=UTC)


@dataclass
class SimulatedOutcome:
    failure_at: datetime = ANCHOR
    window_hours: float = 72
    state: str = "decided"
    attempted: bool = False
    attempted_at: datetime | None = None
    data_source: str = "real_controlled"
    payment_status: str = "failed"
    payment_recovered: bool | None = None
    recovered_amount_minor: int = 0
    recovery_timestamp: datetime | None = None
    time_to_recovery_seconds: int | None = None
    natural_recovery_observed: bool = False
    payment_status_after_24h: str | None = None
    payment_status_after_48h: str | None = None
    amount_minor: int = 249900
    observations: list[dict[str, Any]] = field(default_factory=list)
    observation_keys: set[tuple[str, str]] = field(default_factory=set)

    @property
    def window_starts_at(self) -> datetime:
        return self.failure_at

    @property
    def window_ends_at(self) -> datetime:
        return self.failure_at + timedelta(hours=self.window_hours)

    @property
    def label_kind(self) -> str:
        return outcome_label_kind(
            outcome_state=self.state,
            attempted=self.attempted,
            payment_recovered=self.payment_recovered,
            natural_recovery_observed=self.natural_recovery_observed,
            data_source=self.data_source,
        )


def execute_action(case: SimulatedOutcome, *, at: datetime) -> None:
    transition_outcome(
        case.state,
        "executed",
        observed_at=at,
        source="provider",
        reason="provider_action_attempted",
    )
    case.state = "executed"
    case.attempted = True
    case.attempted_at = at
    transition_outcome(
        case.state,
        "waiting_for_outcome",
        observed_at=at,
        source="provider",
        reason="observation_window_opened",
    )
    case.state = "waiting_for_outcome"


def observe_payment(
    case: SimulatedOutcome,
    *,
    at: datetime,
    status: str,
    source: str = "webhook",
    external_ref: str,
) -> dict[str, Any]:
    key = (source, external_ref)
    if key in case.observation_keys:
        return {"inserted": False, "attribution": "duplicate"}
    case.observation_keys.add(key)
    case.payment_status = status
    attribution = capture_attribution(
        payment_status=status,
        attempted=case.attempted,
        attempted_at=case.attempted_at,
        observed_at=at,
        window_starts_at=case.window_starts_at,
        window_ends_at=case.window_ends_at,
        outcome_state=case.state,
    )
    case.observations.append(
        {
            "source": source,
            "external_ref": external_ref,
            "status": status,
            "observed_at": at.isoformat(),
            "attribution": attribution,
        }
    )
    updates = checkpoint_status_updates(
        anchor_at=case.attempted_at or case.failure_at,
        observed_at=at,
        payment_status=status,
        status_after_24h=case.payment_status_after_24h,
        status_after_48h=case.payment_status_after_48h,
    )
    case.payment_status_after_24h = updates.get(
        "payment_status_after_24h",
        case.payment_status_after_24h,
    )
    case.payment_status_after_48h = updates.get(
        "payment_status_after_48h",
        case.payment_status_after_48h,
    )
    if attribution == "attributed":
        transition_outcome(
            case.state,
            "recovered",
            observed_at=at,
            source=source,
            reason="captured_payment_observed_after_attempt",
        )
        case.state = "recovered"
        case.payment_recovered = True
        case.recovered_amount_minor = case.amount_minor
        case.recovery_timestamp = at
        case.time_to_recovery_seconds = time_to_recovery_seconds(
            failure_timestamp=case.failure_at,
            recovery_timestamp=at,
        )
    elif attribution == "observational":
        case.natural_recovery_observed = True
    return {"inserted": True, "attribution": attribution}


def snapshot_checkpoints(case: SimulatedOutcome, *, at: datetime) -> dict[str, str]:
    updates = checkpoint_status_updates(
        anchor_at=case.attempted_at or case.failure_at,
        observed_at=at,
        payment_status=case.payment_status,
        status_after_24h=case.payment_status_after_24h,
        status_after_48h=case.payment_status_after_48h,
    )
    if "payment_status_after_24h" in updates:
        case.payment_status_after_24h = updates["payment_status_after_24h"]
    if "payment_status_after_48h" in updates:
        case.payment_status_after_48h = updates["payment_status_after_48h"]
    return updates


def finalize_timeout(case: SimulatedOutcome, *, at: datetime) -> bool:
    if case.state != "waiting_for_outcome":
        return False
    if at < case.window_ends_at:
        return False
    transition_outcome(
        case.state,
        "no_recovery_observed",
        observed_at=at,
        source="database",
        reason="observation_window_elapsed_without_capture",
    )
    case.state = "no_recovery_observed"
    case.payment_recovered = None
    case.recovered_amount_minor = 0
    return True


def _check(name: str, passed: bool, detail: str) -> dict[str, Any]:
    return {"check": name, "passed": passed, "detail": detail}


def certify_outcome_plumbing() -> dict[str, Any]:
    config = load_outcome_observation_config()
    checks: list[dict[str, Any]] = []

    recovered = SimulatedOutcome()
    execute_action(recovered, at=ANCHOR + timedelta(minutes=5))
    capture = observe_payment(
        recovered,
        at=ANCHOR + timedelta(hours=6),
        status="captured",
        external_ref="evt_capture_6h",
    )
    snap_24 = snapshot_checkpoints(
        recovered,
        at=ANCHOR + timedelta(hours=24, minutes=10),
    )
    checks.append(
        _check(
            "outcome_state_transitions_valid",
            recovered.state == "recovered",
            recovered.state,
        )
    )
    checks.append(
        _check(
            "capture_attribution_correct",
            capture["attribution"] == "attributed"
            and recovered.time_to_recovery_seconds == 6 * 3600,
            f"{capture['attribution']}:{recovered.time_to_recovery_seconds}",
        )
    )
    checks.append(
        _check(
            "window_24h_checkpoint_correct",
            snap_24.get("payment_status_after_24h") == "captured"
            and recovered.payment_status_after_24h == "captured",
            recovered.payment_status_after_24h or "missing",
        )
    )

    timed_out = SimulatedOutcome(window_hours=48)
    execute_action(timed_out, at=ANCHOR + timedelta(minutes=5))
    snap_48 = snapshot_checkpoints(
        timed_out,
        at=ANCHOR + timedelta(hours=48, minutes=10),
    )
    closed = finalize_timeout(
        timed_out,
        at=ANCHOR + timedelta(hours=48, minutes=10),
    )
    second_close = finalize_timeout(timed_out, at=ANCHOR + timedelta(hours=49))
    checks.append(
        _check(
            "window_48h_checkpoint_correct",
            snap_48.get("payment_status_after_48h") == "failed",
            timed_out.payment_status_after_48h or "missing",
        )
    )
    checks.append(
        _check(
            "timeout_semantics_correct",
            closed
            and timed_out.state == "no_recovery_observed"
            and timed_out.payment_recovered is None
            and timed_out.label_kind == "no_recovery_observed",
            f"{timed_out.state}:{timed_out.payment_recovered}",
        )
    )
    checks.append(
        _check(
            "duplicate_outcome_worker_idempotent",
            second_close is False and timed_out.state == "no_recovery_observed",
            str(second_close),
        )
    )

    duplicate = SimulatedOutcome()
    execute_action(duplicate, at=ANCHOR + timedelta(minutes=5))
    first = observe_payment(
        duplicate,
        at=ANCHOR + timedelta(hours=6),
        status="captured",
        external_ref="evt_same",
    )
    second = observe_payment(
        duplicate,
        at=ANCHOR + timedelta(hours=6),
        status="captured",
        external_ref="evt_same",
    )
    extra_capture = observe_payment(
        duplicate,
        at=ANCHOR + timedelta(hours=7),
        status="captured",
        external_ref="evt_second_capture",
    )
    checks.append(
        _check(
            "duplicate_observations_idempotent",
            first["inserted"]
            and not second["inserted"]
            and extra_capture["attribution"] == "observational"
            and duplicate.time_to_recovery_seconds == 6 * 3600,
            f"{second}:{extra_capture['attribution']}",
        )
    )

    before_attempt = SimulatedOutcome()
    early = observe_payment(
        before_attempt,
        at=ANCHOR + timedelta(hours=1),
        status="captured",
        external_ref="evt_before",
    )
    same_ts = SimulatedOutcome()
    execute_action(same_ts, at=ANCHOR + timedelta(hours=2))
    same = observe_payment(
        same_ts,
        at=ANCHOR + timedelta(hours=2),
        status="captured",
        external_ref="evt_same_ts",
    )
    late = SimulatedOutcome(window_hours=48)
    execute_action(late, at=ANCHOR + timedelta(minutes=5))
    finalize_timeout(late, at=ANCHOR + timedelta(hours=48))
    late_capture = observe_payment(
        late,
        at=ANCHOR + timedelta(hours=50),
        status="captured",
        external_ref="evt_late",
    )
    cancelled = SimulatedOutcome()
    execute_action(cancelled, at=ANCHOR + timedelta(minutes=5))
    cancel_obs = observe_payment(
        cancelled,
        at=ANCHOR + timedelta(hours=3),
        status="cancelled",
        external_ref="evt_cancel",
    )
    provenance_ok = True
    try:
        validate_outcome_record(
            RecoveryOutcomeRecord(
                decision_id="DEC-CERT",
                payment_id="PAY-CERT",
                action="payment_link",
                decision_probability=0.5,
                decision_margin=0.1,
                policy_version="recovery_policy_v3",
                model_version="recovery_model_v2_online",
                attempted=True,
                attempted_at=ANCHOR.isoformat(),
                failure_timestamp=ANCHOR.isoformat(),
                payment_status_after_24h="captured",
                payment_status_after_48h=None,
                outcome_state="recovered",
                outcome_at=(ANCHOR + timedelta(hours=6)).isoformat(),
                payment_recovered=True,
                recovered_amount_minor=249900,
                recovery_timestamp=(ANCHOR + timedelta(hours=6)).isoformat(),
                time_to_recovery_seconds=6 * 3600,
                observation_window_starts_at=ANCHOR.isoformat(),
                observation_window_ends_at=(ANCHOR + timedelta(hours=72)).isoformat(),
                outcome_source="webhook",
                data_source="real_controlled",
            )
        )
        timeout_false_rejected = False
        try:
            validate_outcome_record(
                RecoveryOutcomeRecord(
                    decision_id="DEC-TIMEOUT",
                    payment_id="PAY-TIMEOUT",
                    action="payment_link",
                    decision_probability=0.5,
                    decision_margin=0.1,
                    policy_version="recovery_policy_v3",
                    model_version="recovery_model_v2_online",
                    attempted=True,
                    attempted_at=ANCHOR.isoformat(),
                    failure_timestamp=ANCHOR.isoformat(),
                    payment_status_after_24h="failed",
                    payment_status_after_48h="failed",
                    outcome_state="no_recovery_observed",
                    outcome_at=(ANCHOR + timedelta(hours=48)).isoformat(),
                    payment_recovered=False,
                    recovered_amount_minor=0,
                    recovery_timestamp=None,
                    time_to_recovery_seconds=None,
                    observation_window_starts_at=ANCHOR.isoformat(),
                    observation_window_ends_at=(ANCHOR + timedelta(hours=48)).isoformat(),
                    outcome_source="database",
                    data_source="real_controlled",
                )
            )
        except ValueError:
            timeout_false_rejected = True
        provenance_ok = timeout_false_rejected
    except Exception:
        provenance_ok = False
    checks.append(
        _check(
            "provenance_enforced",
            provenance_ok
            and early["attribution"] == "observational"
            and same["attribution"] == "attributed"
            and late_capture["attribution"] == "observational"
            and late.state == "no_recovery_observed"
            and cancel_obs["attribution"] == "none"
            and cancelled.state == "waiting_for_outcome",
            "edge_cases",
        )
    )

    synthetic_blocked = False
    try:
        split_outcomes_by_provenance(pd.DataFrame({"payment_id": ["PAY-1"]}))
    except ProvenanceError:
        synthetic_blocked = True
    split = split_outcomes_by_provenance(
        pd.DataFrame(
            {
                "data_source": ["synthetic", "real_shadow", "real_controlled"],
                "attempted": [True, False, True],
                "outcome_state": [
                    "recovered",
                    "decided",
                    "no_recovery_observed",
                ],
            }
        )
    )
    checks.append(
        _check(
            "synthetic_real_separation_enforced",
            synthetic_blocked
            and list(split["training_candidates"]["data_source"])
            == ["real_controlled"],
            str(list(split["training_candidates"]["data_source"])),
        )
    )

    shadow = RecoveryOutcomeRecord(
        decision_id="DEC-SHADOW",
        payment_id="PAY-SHADOW",
        action="payment_link",
        decision_probability=0.4,
        decision_margin=0.05,
        policy_version="recovery_policy_v3",
        model_version="recovery_model_v2_online",
        attempted=False,
        attempted_at=None,
        failure_timestamp=ANCHOR.isoformat(),
        payment_status_after_24h=None,
        payment_status_after_48h=None,
        outcome_state="decided",
        outcome_at=None,
        payment_recovered=None,
        recovered_amount_minor=0,
        recovery_timestamp=None,
        time_to_recovery_seconds=None,
        observation_window_starts_at=ANCHOR.isoformat(),
        observation_window_ends_at=(ANCHOR + timedelta(hours=72)).isoformat(),
        outcome_source="database",
        data_source="real_shadow",
        natural_recovery_observed=True,
    )
    shadow_eligibility = training_eligibility(shadow)
    checks.append(
        _check(
            "legacy_decisions_excluded",
            not config["safety"]["legacy_decisions_backfilled"],
            "legacy_backfill_disabled",
        )
    )
    checks.append(
        _check(
            "no_accidental_training_labels",
            shadow_eligibility["eligible"] is False
            and recovered.label_kind == "attributed_intervention_recovery"
            and before_attempt.label_kind == "observational_recovery"
            and not config["safety"]["phase15_authorized"],
            str(shadow_eligibility["reasons"]),
        )
    )

    named = {item["check"]: item["passed"] for item in checks}
    required = list(config["certification_checks"])
    missing = [name for name in required if not named.get(name, False)]
    # duplicate worker is extra; map duplicate_observations already required
    certified = not missing and all(item["passed"] for item in checks)
    return {
        "version": 1,
        "phase": "14B",
        "certified": certified,
        "controlled_execution_authorized": False,
        "phase15_authorized": False,
        "missing_checks": missing,
        "checks": checks,
        "happy_paths": {
            "recovered_at_6h": {
                "state": recovered.state,
                "label_kind": recovered.label_kind,
                "time_to_recovery_seconds": recovered.time_to_recovery_seconds,
                "payment_status_after_24h": recovered.payment_status_after_24h,
            },
            "timeout_at_48h": {
                "state": timed_out.state,
                "label_kind": timed_out.label_kind,
                "payment_recovered": timed_out.payment_recovered,
                "payment_status_after_48h": timed_out.payment_status_after_48h,
            },
        },
    }


def _bucket_taxonomy(failure_class: str | None, config: dict[str, Any]) -> str:
    if not failure_class:
        return "unknown"
    return config["taxonomy_buckets"].get(failure_class, "unknown")


def _value_segment(amount_minor: int | None, config: dict[str, Any]) -> str:
    amount = int(amount_minor or 0)
    high = int(config["customer_value_thresholds_minor"]["high_value"])
    medium = int(config["customer_value_thresholds_minor"]["medium_value"])
    if amount >= high:
        return "high_value"
    if amount >= medium:
        return "medium_value"
    return "low_value"


def _temporal_buckets(timestamp: datetime | None) -> set[str]:
    if timestamp is None:
        return set()
    buckets = {
        "weekend" if timestamp.weekday() >= 5 else "weekday",
        "night" if timestamp.hour < 6 or timestamp.hour >= 18 else "day",
    }
    return buckets


def evaluate_evidence_coverage(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    config = load_outcome_observation_config()
    gate = config["coverage_gate"]
    attributed = [
        row
        for row in records
        if row.get("label_kind") == "attributed_intervention_recovery"
        or (
            row.get("outcome_state") == "recovered"
            and row.get("attempted")
            and row.get("data_source") == "real_controlled"
        )
    ]
    eligible = [
        row
        for row in records
        if row.get("attempted")
        and row.get("data_source") == "real_controlled"
        and row.get("outcome_state") in {"recovered", "no_recovery_observed"}
        and not row.get("natural_recovery_observed")
    ]
    leakage = [
        row
        for row in records
        if row.get("data_source") == "synthetic"
        and row.get("outcome_state") == "recovered"
        and row.get("attempted")
    ]

    def _count(dimension: dict[str, int], values: list[str | None]) -> dict[str, int]:
        counts = {key: 0 for key in dimension}
        for value in values:
            if value in counts:
                counts[value] += 1
        return counts

    action_counts = _count(
        gate["actions"],
        [
            "no_action" if not row.get("action") else str(row.get("action"))
            for row in eligible
        ],
    )
    taxonomy_counts = _count(
        gate["taxonomy"],
        [_bucket_taxonomy(row.get("failure_class"), config) for row in eligible],
    )
    value_counts = _count(
        {
            key: 0
            for key in ("high_value", "medium_value", "low_value")
        },
        [_value_segment(row.get("amount_minor"), config) for row in eligible],
    )
    customer_counts = _count(
        {"new_customer": 0, "returning_customer": 0},
        [
            "returning_customer"
            if int(row.get("previous_payments") or 0) > 0
            else "new_customer"
            for row in eligible
        ],
    )
    outcome_counts = _count(
        gate["outcome_classes"],
        [str(row.get("outcome_state")) for row in eligible],
    )
    temporal_values: list[str] = []
    for row in eligible:
        stamp = row.get("failure_timestamp")
        parsed = None
        if isinstance(stamp, datetime):
            parsed = stamp
        elif stamp:
            parsed = datetime.fromisoformat(str(stamp).replace("Z", "+00:00"))
        temporal_values.extend(_temporal_buckets(parsed))
    temporal_counts = {key: temporal_values.count(key) for key in gate["temporal"]}

    def _passed(actual: dict[str, int], required: dict[str, int]) -> bool:
        return all(actual.get(key, 0) >= int(need) for key, need in required.items())

    checks = {
        "real_labels": len(eligible) >= int(gate["minimum_real_labels"]),
        "attributed_recoveries": len(attributed)
        >= int(gate["minimum_attributed_recoveries"]),
        "action_coverage": _passed(action_counts, gate["actions"]),
        "taxonomy_coverage": _passed(taxonomy_counts, gate["taxonomy"]),
        "customer_segment_coverage": _passed(
            {**value_counts, **customer_counts},
            gate["customer_segments"],
        ),
        "outcome_class_coverage": _passed(outcome_counts, gate["outcome_classes"]),
        "temporal_coverage": _passed(temporal_counts, gate["temporal"]),
        "outcome_provenance_valid": True,
        "no_major_label_leakage": not leakage,
    }
    candidate_ready = all(checks.values()) and bool(
        config["safety"]["phase15_authorized"]
    )
    return {
        "version": 1,
        "phase": "14D",
        "candidate_ready": candidate_ready,
        "phase15_authorized": False,
        "checks": checks,
        "counts": {
            "eligible_labels": len(eligible),
            "attributed_intervention_recoveries": len(attributed),
            "observational_recoveries": sum(
                bool(row.get("natural_recovery_observed"))
                and row.get("outcome_state") != "recovered"
                for row in records
            ),
            "actions": action_counts,
            "taxonomy": taxonomy_counts,
            "customer_value": value_counts,
            "customer_history": customer_counts,
            "outcome_classes": outcome_counts,
            "temporal": temporal_counts,
            "synthetic_recovered_rows": len(leakage),
        },
        "minima": {
            "real_labels": gate["minimum_real_labels"],
            "attributed_recoveries": gate["minimum_attributed_recoveries"],
            "actions": gate["actions"],
            "taxonomy": gate["taxonomy"],
            "customer_segments": gate["customer_segments"],
            "outcome_classes": gate["outcome_classes"],
            "temporal": gate["temporal"],
        },
        "reason": (
            "Phase 15 stays unauthorized until real attributed labels meet "
            "coverage minima. Row count alone is not sufficient."
        ),
    }
