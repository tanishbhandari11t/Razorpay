from __future__ import annotations

"""
Final challenger trainer entrypoint.

Refuses to train while Phase 15 is unauthorized. Writes a blocked model card
and readiness report instead of mutating frozen V1/V2 artifacts.
"""

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from ml.src.final_evidence_gate import evaluate_final_evidence_gate
from ml.src.train_recovery_model_v3_challenger import (
    ChallengerBlockedError,
    assert_training_authorized,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "final"
ARTIFACT_DIR = REPO_ROOT / "ml" / "artifacts" / "final"
MODEL_CARD_PATH = REPORT_DIR / "model_card.json"
READINESS_PATH = REPORT_DIR / "challenger_readiness.json"
V3_CONFIG = REPO_ROOT / "ml" / "config" / "recovery_model_v3_challenger.yaml"


def write_blocked_model_card(
    *,
    evidence_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    gate = evaluate_final_evidence_gate(evidence_records or [])
    config = yaml.safe_load(V3_CONFIG.read_text(encoding="utf-8"))
    blocked_reason = None
    try:
        assert_training_authorized(evidence_records or [])
    except ChallengerBlockedError as exc:
        blocked_reason = str(exc)
    card = {
        "version": 1,
        "model_version": config["identity"]["model_version"],
        "status": "blocked",
        "model_ready": False,
        "phase15_authorized": False,
        "controlled_execution_authorized": False,
        "training_data": {
            "allowed_sources": ["real_controlled", "synthetic"],
            "require_data_source_column": True,
            "observational_as_attributed": False,
            "current_eligible_rows": gate["counts"][
                "attributed_intervention_recoveries"
            ],
        },
        "features": {
            "mode": "action_conditioned",
            "objective": "P(recovery | features, action)",
        },
        "limitations": [
            "Insufficient real attributed intervention labels",
            "Shadow observational recoveries are not causal uplift",
            "V1 and V2-online remain the frozen baselines",
        ],
        "metrics": None,
        "known_failure_modes": [
            "weak_decision_margin",
            "unknown_taxonomy",
            "feature_support_below_threshold",
        ],
        "deployment_constraints": {
            "execution_mode": "shadow",
            "provider_actions_enabled": False,
            "qwen_tools_enabled": False,
        },
        "evidence_gate": gate,
        "blocked_reason": blocked_reason
        or "Final challenger training remains unauthorized",
        "generated_at": datetime.now(UTC).isoformat(),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_CARD_PATH.write_text(
        json.dumps(card, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    readiness = {
        "version": 1,
        "model_ready": False,
        "phase15_authorized": False,
        "candidate_ready": gate["candidate_ready"],
        "blocked_reason": card["blocked_reason"],
        "artifact_dir": str(ARTIFACT_DIR.relative_to(REPO_ROOT)).replace(
            "\\", "/"
        ),
        "generated_at": datetime.now(UTC).isoformat(),
    }
    READINESS_PATH.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "README.txt").write_text(
        "Final challenger artifacts are intentionally empty until Phase 15 "
        "is authorized by the evidence gate.\n",
        encoding="utf-8",
    )
    return card


def main() -> None:
    card = write_blocked_model_card([])
    print(json.dumps(card, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
