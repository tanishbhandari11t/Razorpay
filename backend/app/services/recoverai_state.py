from __future__ import annotations

"""
RecoverAI final-phase orchestration state machine.

States only advance when evidence/safety gates pass. Manual flag flips cannot
skip OBSERVATION → MODEL_READY.
"""

import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.evidence_coverage import build_evidence_coverage_report
from ml.src.observe_recovery import load_outcome_observation_config


class RecoverAIState(StrEnum):
    OBSERVATION = "OBSERVATION"
    EVIDENCE_INSUFFICIENT = "EVIDENCE_INSUFFICIENT"
    EVIDENCE_READY = "EVIDENCE_READY"
    CHALLENGER_EVALUATION = "CHALLENGER_EVALUATION"
    MODEL_READY = "MODEL_READY"
    QWEN_READY = "QWEN_READY"
    CONTROLLED_READY = "CONTROLLED_READY"
    PILOT = "PILOT"
    FINAL = "FINAL"


@dataclass(frozen=True)
class RecoverAIProgress:
    state: RecoverAIState
    phase15_authorized: bool
    model_ready: bool
    qwen_ready: bool
    controlled_ready: bool
    pilot_active: bool
    final_ready: bool
    blockers: list[str]
    evidence: dict[str, Any]
    next_action: str


def evaluate_recoverai_state(
    *,
    evidence_records: list[dict[str, Any]] | None = None,
    challenger_passed: bool = False,
    safety_evaluation_passed: bool = False,
    qwen_validated: bool = False,
    controlled_gates_passed: bool = False,
    pilot_completed: bool = False,
    final_evaluation_complete: bool = False,
    execution_mode: str = "shadow",
    training_eligible_labels: int = 0,
    shadow_decisions: int = 0,
    real_failures_observed: int = 0,
) -> RecoverAIProgress:
    records = evidence_records or []
    coverage = build_evidence_coverage_report(records)
    observation = load_outcome_observation_config()
    real_cases = max(
        int(coverage.get("real_cases", len(records))),
        int(shadow_decisions),
        int(real_failures_observed),
    )
    candidate_ready = bool(coverage.get("candidate_ready"))
    # Phase 15 cannot self-authorize from coverage alone.
    phase15_authorized = bool(
        observation["safety"]["phase15_authorized"]
    ) and candidate_ready
    eligible = int(training_eligible_labels)
    blockers: list[str] = []

    if execution_mode != "shadow" and not controlled_gates_passed:
        blockers.append("execution_mode_not_shadow_without_gates")
    if not candidate_ready:
        blockers.append("evidence_coverage_incomplete")
    if eligible <= 0:
        blockers.append("no_attributed_training_labels")
    if not phase15_authorized:
        blockers.append("phase15_unauthorized")
    if not challenger_passed:
        blockers.append("challenger_not_evaluated")
    if not safety_evaluation_passed:
        blockers.append("safety_evaluation_incomplete")
    if not qwen_validated:
        blockers.append("qwen_not_validated")
    if not controlled_gates_passed:
        blockers.append("controlled_gates_incomplete")
    if not pilot_completed:
        blockers.append("pilot_not_completed")
    if not final_evaluation_complete:
        blockers.append("final_evaluation_incomplete")

    if real_cases == 0:
        state = RecoverAIState.OBSERVATION
        next_action = (
            "Collect Razorpay Test Mode failures and let the observer run."
        )
    elif not candidate_ready or not phase15_authorized:
        state = RecoverAIState.EVIDENCE_INSUFFICIENT
        next_action = (
            "Keep shadow collection running until coverage minima are met "
            "and Phase 15 is legitimately authorized."
        )
    elif not challenger_passed:
        state = RecoverAIState.EVIDENCE_READY
        next_action = (
            "Train/evaluate the final challenger against frozen baselines."
        )
    elif not safety_evaluation_passed:
        state = RecoverAIState.CHALLENGER_EVALUATION
        next_action = "Complete safety evaluation before model_ready."
    elif not qwen_validated:
        state = RecoverAIState.MODEL_READY
        next_action = "Validate Qwen fail-closed communication layer."
    elif not controlled_gates_passed:
        state = RecoverAIState.QWEN_READY
        next_action = "Pass controlled-execution allowlist and limit gates."
    elif not pilot_completed:
        state = RecoverAIState.CONTROLLED_READY
        next_action = "Run a tiny payment_link-only controlled pilot."
    elif not final_evaluation_complete:
        state = RecoverAIState.PILOT
        next_action = (
            "Generate final evaluation, dashboard, and demo narrative."
        )
    else:
        state = RecoverAIState.FINAL
        next_action = "RecoverAI final phase complete."

    model_ready = (
        challenger_passed
        and safety_evaluation_passed
        and state
        in {
            RecoverAIState.MODEL_READY,
            RecoverAIState.QWEN_READY,
            RecoverAIState.CONTROLLED_READY,
            RecoverAIState.PILOT,
            RecoverAIState.FINAL,
        }
    )
    return RecoverAIProgress(
        state=state,
        phase15_authorized=phase15_authorized,
        model_ready=model_ready,
        qwen_ready=qwen_validated
        and state
        in {
            RecoverAIState.QWEN_READY,
            RecoverAIState.CONTROLLED_READY,
            RecoverAIState.PILOT,
            RecoverAIState.FINAL,
        },
        controlled_ready=controlled_gates_passed
        and state
        in {
            RecoverAIState.CONTROLLED_READY,
            RecoverAIState.PILOT,
            RecoverAIState.FINAL,
        },
        pilot_active=(
            controlled_gates_passed
            and not pilot_completed
            and state == RecoverAIState.CONTROLLED_READY
        ),
        final_ready=state == RecoverAIState.FINAL,
        blockers=blockers,
        evidence={
            "real_cases": real_cases,
            "candidate_ready": candidate_ready,
            "phase15_authorized": phase15_authorized,
            "coverage_checks": coverage.get("checks", {}),
            "observational_recoveries": coverage.get(
                "observational_recoveries", 0
            ),
            "attributed_intervention_recoveries": coverage.get(
                "attributed_intervention_recoveries", 0
            ),
            "training_eligible_labels": eligible,
        },
        next_action=next_action,
    )


def progress_to_dict(progress: RecoverAIProgress) -> dict[str, Any]:
    return {
        "state": progress.state.value,
        "phase15_authorized": progress.phase15_authorized,
        "model_ready": progress.model_ready,
        "qwen_ready": progress.qwen_ready,
        "controlled_ready": progress.controlled_ready,
        "pilot_active": progress.pilot_active,
        "final_ready": progress.final_ready,
        "blockers": progress.blockers,
        "evidence": progress.evidence,
        "next_action": progress.next_action,
        "execution_mode": "shadow",
        "manual_skip_forbidden": True,
    }
