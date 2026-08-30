from __future__ import annotations

from datetime import UTC, datetime, timedelta
import hashlib
import json
from pathlib import Path

import pytest

from ml.src.observe_recovery import (
    checkpoint_status_updates,
    observed_terminal_state,
    time_to_recovery_seconds,
    transition_outcome,
)
from ml.src.build_real_outcome_dataset import (
    ProvenanceError,
    split_outcomes_by_provenance,
)
from ml.src.record_recovery_outcome import (
    RecoveryOutcomeRecord,
    load_real_outcome_schema,
    retraining_gate,
    training_eligibility,
    validate_outcome_record,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _record(**overrides) -> RecoveryOutcomeRecord:
    values = {
        "decision_id": "DEC-1",
        "payment_id": "PAY-1",
        "action": "payment_link",
        "decision_probability": 0.7,
        "decision_margin": 0.1,
        "policy_version": "recovery_policy_v3",
        "model_version": "recovery_model_v2_online",
        "attempted": False,
        "attempted_at": None,
        "failure_timestamp": "2026-01-01T00:00:00+00:00",
        "payment_status_after_24h": None,
        "payment_status_after_48h": None,
        "outcome_state": "decided",
        "outcome_at": None,
        "payment_recovered": None,
        "recovered_amount_minor": 0,
        "recovery_timestamp": None,
        "time_to_recovery_seconds": None,
        "observation_window_starts_at": "2026-01-01T00:00:00+00:00",
        "observation_window_ends_at": "2026-01-04T00:00:00+00:00",
        "outcome_source": "database",
        "data_source": "real_shadow",
    }
    values.update(overrides)
    return RecoveryOutcomeRecord(**values)


def test_phase14_schema_cannot_enable_execution() -> None:
    schema = load_real_outcome_schema()
    assert schema["safety"]["controlled_execution_authorized"] is False
    assert schema["safety"]["provider_actions_enabled"] is False
    assert schema["training_eligibility"]["allow_real_shadow"] is False


def test_outcome_state_machine_rejects_skipped_transition() -> None:
    with pytest.raises(ValueError, match="Invalid outcome transition"):
        transition_outcome(
            "decided",
            "recovered",
            observed_at=datetime.now(UTC),
            source="webhook",
            reason="capture",
        )


def test_terminal_outcome_is_immutable() -> None:
    with pytest.raises(ValueError, match="immutable"):
        transition_outcome(
            "recovered",
            "unknown",
            observed_at=datetime.now(UTC),
            source="database",
            reason="conflict",
        )


def test_unattempted_shadow_recovery_cannot_be_a_label() -> None:
    record = _record(
        outcome_state="recovered",
        payment_recovered=True,
        recovered_amount_minor=249900,
    )
    with pytest.raises(ValueError, match="Unattempted"):
        validate_outcome_record(record)


def test_real_shadow_decision_is_not_training_eligible() -> None:
    result = training_eligibility(_record())
    assert result["eligible"] is False
    assert "outcome_not_terminal" in result["reasons"]
    assert "action_not_attempted" in result["reasons"]
    assert "data_source_not_real_controlled" in result["reasons"]


def test_timeout_means_no_recovery_observed() -> None:
    now = datetime.now(UTC)
    assert (
        observed_terminal_state(
            payment_status="failed",
            attempted=True,
            observed_at=now,
            observation_window_ends_at=now - timedelta(seconds=1),
        )
        == "no_recovery_observed"
    )


def test_capture_requires_attempt_and_window() -> None:
    now = datetime.now(UTC)
    assert (
        observed_terminal_state(
            payment_status="captured",
            attempted=False,
            observed_at=now,
            observation_window_ends_at=now + timedelta(hours=1),
        )
        is None
    )
    assert (
        observed_terminal_state(
            payment_status="captured",
            attempted=True,
            observed_at=now,
            observation_window_ends_at=now + timedelta(hours=1),
        )
        == "recovered"
    )


def test_time_to_recovery_uses_failure_timestamp() -> None:
    failed_at = datetime(2026, 1, 1, tzinfo=UTC)
    recovered_at = failed_at + timedelta(hours=5)
    assert time_to_recovery_seconds(
        failure_timestamp=failed_at,
        recovery_timestamp=recovered_at,
    ) == 18000


def test_status_checkpoints_fill_after_24h_and_48h() -> None:
    anchor = datetime(2026, 1, 1, tzinfo=UTC)
    assert checkpoint_status_updates(
        anchor_at=anchor,
        observed_at=anchor + timedelta(hours=10),
        payment_status="failed",
        status_after_24h=None,
        status_after_48h=None,
    ) == {}
    assert checkpoint_status_updates(
        anchor_at=anchor,
        observed_at=anchor + timedelta(hours=25),
        payment_status="failed",
        status_after_24h=None,
        status_after_48h=None,
    ) == {"payment_status_after_24h": "failed"}
    assert checkpoint_status_updates(
        anchor_at=anchor,
        observed_at=anchor + timedelta(hours=49),
        payment_status="captured",
        status_after_24h="failed",
        status_after_48h=None,
    ) == {"payment_status_after_48h": "captured"}


def test_retraining_stays_blocked_below_coverage_thresholds() -> None:
    gate = retraining_gate(20, coverage_passed=False)
    assert gate["stage"] == "no_retraining"
    assert gate["authorized"] is False
    experimental = retraining_gate(150, coverage_passed=True)
    assert experimental["stage"] == "experimental_challenger"
    assert experimental["authorized"] is False


def test_outcome_dataset_refuses_unlabeled_mix() -> None:
    import pandas as pd

    with pytest.raises(ProvenanceError, match="data_source"):
        split_outcomes_by_provenance(
            pd.DataFrame({"payment_id": ["PAY-1"], "outcome_state": ["recovered"]})
        )
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
    assert list(split["synthetic"]["data_source"]) == ["synthetic"]
    assert list(split["real"]["data_source"]) == [
        "real_shadow",
        "real_controlled",
    ]
    assert list(split["training_candidates"]["data_source"]) == [
        "real_controlled"
    ]


def test_timeout_cannot_be_labeled_recovered_false() -> None:
    record = _record(
        attempted=True,
        attempted_at="2026-01-01T01:00:00+00:00",
        outcome_state="no_recovery_observed",
        payment_recovered=False,
        data_source="real_controlled",
    )
    with pytest.raises(ValueError, match="recovered=false"):
        validate_outcome_record(record)


def test_phase14b_certifies_plumbing_and_keeps_phase15_blocked() -> None:
    from ml.src.validate_outcomes import (
        certify_outcome_plumbing,
        evaluate_evidence_coverage,
    )

    report = certify_outcome_plumbing()
    assert report["certified"] is True
    assert report["phase15_authorized"] is False
    assert report["controlled_execution_authorized"] is False
    assert report["happy_paths"]["recovered_at_6h"]["time_to_recovery_seconds"] == 21600
    assert report["happy_paths"]["timeout_at_48h"]["payment_recovered"] is None
    coverage = evaluate_evidence_coverage([])
    assert coverage["candidate_ready"] is False
    assert coverage["phase15_authorized"] is False
    assert coverage["checks"]["real_labels"] is False


def test_v3_challenger_refuses_unauthorized_training() -> None:
    from ml.src.train_recovery_model_v3_challenger import (
        ChallengerBlockedError,
        assert_training_authorized,
        write_phase15_readiness,
    )

    with pytest.raises(ChallengerBlockedError, match="blocked"):
        assert_training_authorized([])
    readiness = write_phase15_readiness([])
    assert readiness["model_ready"] is False
    assert readiness["phase15_authorized"] is False


def test_phase14_manifest_preserves_evidence_boundaries() -> None:
    report_dir = REPO_ROOT / "ml" / "reports" / "phase14"
    manifest = json.loads(
        (report_dir / "phase14_manifest.json").read_text(encoding="utf-8")
    )
    summary = json.loads(
        (report_dir / "phase14_summary.json").read_text(encoding="utf-8")
    )
    validation = json.loads(
        (report_dir / "outcome_validation.json").read_text(encoding="utf-8")
    )
    coverage = json.loads(
        (report_dir / "outcome_coverage.json").read_text(encoding="utf-8")
    )
    coverage_alias = json.loads(
        (report_dir / "coverage.json").read_text(encoding="utf-8")
    )
    phase15 = json.loads(
        (
            REPO_ROOT / "ml" / "reports" / "phase15" / "phase15_readiness.json"
        ).read_text(encoding="utf-8")
    )
    for path, entry in {
        **manifest["inputs"],
        **manifest["artifacts"],
    }.items():
        actual = hashlib.sha256((REPO_ROOT / path).read_bytes()).hexdigest()
        assert actual == entry["sha256"]
    assert manifest["controlled_execution_authorized"] is False
    assert manifest["retraining_authorized"] is False
    assert manifest["phase15_authorized"] is False
    assert summary["training_readiness"]["retraining_allowed"] is False
    assert summary["training_readiness"]["retraining_stage"] == "no_retraining"
    assert summary["evidence_inventory"]["attributed_intervention_recoveries"] == 0
    assert summary["evidence_inventory"]["training_eligible_labels"] == 0
    assert summary["evidence_boundaries"][
        "real_shadow_outcomes_are_causal_uplift"
    ] is False
    assert summary["evidence_boundaries"]["dataset_split"][
        "mixed_without_provenance"
    ] is False
    assert validation["certified"] is True
    assert coverage["candidate_ready"] is False
    assert coverage["phase15_authorized"] is False
    assert coverage_alias["phase15_authorized"] is False
    assert phase15["model_ready"] is False
    assert phase15["phase15_authorized"] is False
