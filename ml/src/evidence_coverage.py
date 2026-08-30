from __future__ import annotations

"""
Phase 14D evidence coverage gate.

Counts observational vs attributed labels separately and refuses to authorize
Phase 15 until coverage minima are met. Row count alone is never enough.
"""

import json
from pathlib import Path
from typing import Any

from ml.src.observe_recovery import load_outcome_observation_config
from ml.src.validate_outcomes import evaluate_evidence_coverage


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase14"
COVERAGE_PATH = REPORT_DIR / "coverage.json"
OUTCOME_COVERAGE_PATH = REPORT_DIR / "outcome_coverage.json"


def build_evidence_coverage_report(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = evaluate_evidence_coverage(records)
    config = load_outcome_observation_config()
    observational = sum(
        1
        for row in records
        if row.get("label_kind") == "observational_recovery"
        or (
            row.get("natural_recovery_observed")
            and row.get("payment_recovered") is not True
        )
    )
    no_recovery = sum(
        1
        for row in records
        if row.get("outcome_state") == "no_recovery_observed"
    )
    unknown = sum(
        1 for row in records if row.get("outcome_state") == "unknown"
    )
    attributed = coverage["counts"]["attributed_intervention_recoveries"]
    report = {
        "version": 1,
        "phase": "14D",
        "real_cases": len(records),
        "observational_recoveries": observational,
        "attributed_intervention_recoveries": attributed,
        "no_recovery_observed": no_recovery,
        "unknown": unknown,
        "action_coverage": coverage["counts"]["actions"],
        "taxonomy_coverage": coverage["counts"]["taxonomy"],
        "segment_coverage": {
            **coverage["counts"]["customer_value"],
            **coverage["counts"]["customer_history"],
        },
        "temporal_coverage": coverage["counts"]["temporal"],
        "outcome_coverage": coverage["counts"]["outcome_classes"],
        "checks": coverage["checks"],
        "minima": coverage["minima"],
        "candidate_ready": coverage["candidate_ready"],
        "phase15_authorized": False,
        "safety": {
            "controlled_execution_authorized": False,
            "provider_actions_enabled": False,
            "observational_equals_attributed": False,
            **config["safety"],
        },
        "reason": coverage["reason"],
    }
    return report


def write_evidence_coverage_report(
    records: list[dict[str, Any]],
    *,
    path: Path = COVERAGE_PATH,
) -> dict[str, Any]:
    report = build_evidence_coverage_report(records)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(report, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    # Keep the Phase 14B artifact name in sync for existing tests.
    OUTCOME_COVERAGE_PATH.write_text(payload, encoding="utf-8")
    return report


def phase15_authorization(records: list[dict[str, Any]]) -> dict[str, Any]:
    report = build_evidence_coverage_report(records)
    return {
        "authorized": False,
        "candidate_ready": report["candidate_ready"],
        "checks": report["checks"],
        "reason": (
            "Phase 15 stays locked until evidence coverage authorizes it. "
            "This function never flips authorized=true from Phase 14 alone."
        ),
    }
