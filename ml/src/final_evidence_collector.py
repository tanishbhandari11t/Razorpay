from __future__ import annotations

"""
Final evidence inventory and continuous coverage reporter.

Does not unlock Phase 15. Does not execute actions. Reports observational vs
attributed recoveries separately.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from ml.src.evidence_coverage import build_evidence_coverage_report
from ml.src.final_evidence_gate import evaluate_final_evidence_gate


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "final"
INVENTORY_PATH = REPORT_DIR / "evidence_inventory.json"


def build_evidence_inventory(
    records: list[dict[str, Any]],
    *,
    shadow_decisions: int | None = None,
    real_actions_executed: int = 0,
) -> dict[str, Any]:
    coverage = build_evidence_coverage_report(records)
    gate = evaluate_final_evidence_gate(records)
    inventory = {
        "version": 1,
        "generated_at": datetime.now(UTC).isoformat(),
        "real_cases": max(
            coverage["real_cases"],
            shadow_decisions or 0,
        ),
        "tracked_outcomes": coverage["real_cases"],
        "shadow_decisions": (
            shadow_decisions
            if shadow_decisions is not None
            else coverage["real_cases"]
        ),
        "real_actions_executed": real_actions_executed,
        "observational_recoveries": coverage["observational_recoveries"],
        "attributed_intervention_recoveries": coverage[
            "attributed_intervention_recoveries"
        ],
        "no_recovery_observed": coverage["no_recovery_observed"],
        "unknown": coverage["unknown"],
        "training_eligible_labels": int(
            coverage["checks"].get("real_labels") is True
            and coverage.get("attributed_intervention_recoveries", 0)
            or 0
        ),
        "actions": coverage["action_coverage"],
        "taxonomy": coverage["taxonomy_coverage"],
        "segments": coverage["segment_coverage"],
        "temporal": coverage["temporal_coverage"],
        "outcome_coverage": coverage["outcome_coverage"],
        "phase15_authorized": False,
        "final_evidence_gate": gate,
        "note": (
            "Shadow decisions are not training examples. Observational "
            "recoveries are not attributed intervention recoveries."
        ),
    }
    # Prefer explicit eligible count from attributed terminal controlled labels.
    inventory["training_eligible_labels"] = sum(
        1
        for row in records
        if row.get("attempted")
        and row.get("data_source") == "real_controlled"
        and row.get("outcome_state") in {"recovered", "no_recovery_observed"}
        and not (
            row.get("natural_recovery_observed")
            and row.get("outcome_state") != "recovered"
        )
    )
    return inventory


def write_evidence_inventory(
    records: list[dict[str, Any]],
    *,
    path: Path = INVENTORY_PATH,
    shadow_decisions: int | None = None,
    real_actions_executed: int = 0,
) -> dict[str, Any]:
    inventory = build_evidence_inventory(
        records,
        shadow_decisions=shadow_decisions,
        real_actions_executed=real_actions_executed,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(inventory, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return inventory
