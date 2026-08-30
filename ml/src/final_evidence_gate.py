from __future__ import annotations

"""
Final evidence gate for Phase 15 authorization.

Never self-authorizes: phase15_authorized remains false until an explicit
later safety flip after candidate_ready is true. This module only reports.
"""

from typing import Any

from ml.src.evidence_coverage import build_evidence_coverage_report
from ml.src.observe_recovery import load_outcome_observation_config


def evaluate_final_evidence_gate(
    records: list[dict[str, Any]],
) -> dict[str, Any]:
    coverage = build_evidence_coverage_report(records)
    observation = load_outcome_observation_config()
    checks = dict(coverage["checks"])
    duplicate_ids = [
        row.get("decision_id")
        for row in records
        if row.get("decision_id")
    ]
    checks["zero_duplicate_decision_ids"] = len(duplicate_ids) == len(
        set(duplicate_ids)
    )
    checks["observational_not_mixed_as_attributed"] = all(
        not (
            row.get("natural_recovery_observed")
            and row.get("payment_recovered") is True
            and not row.get("attempted")
        )
        for row in records
    )
    checks["config_phase15_still_locked"] = not bool(
        observation["safety"]["phase15_authorized"]
    )
    candidate_ready = all(
        value
        for key, value in checks.items()
        if key != "config_phase15_still_locked"
    )
    return {
        "version": 1,
        "candidate_ready": candidate_ready and bool(coverage["candidate_ready"]),
        "phase15_authorized": False,
        "checks": checks,
        "minima": coverage["minima"],
        "counts": {
            "real_cases": coverage["real_cases"],
            "observational_recoveries": coverage["observational_recoveries"],
            "attributed_intervention_recoveries": coverage[
                "attributed_intervention_recoveries"
            ],
            "no_recovery_observed": coverage["no_recovery_observed"],
            "unknown": coverage["unknown"],
            "actions": coverage["action_coverage"],
            "taxonomy": coverage["taxonomy_coverage"],
            "segments": coverage["segment_coverage"],
            "temporal": coverage["temporal_coverage"],
        },
        "reason": (
            "Final evidence gate never auto-authorizes Phase 15. "
            "Keep collecting until candidate_ready is true, then an explicit "
            "human/safety review may unlock training."
        ),
    }
