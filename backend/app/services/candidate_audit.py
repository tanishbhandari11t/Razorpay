from __future__ import annotations

"""
Shadow candidate audit trail.

Records all candidate action scores even when nothing is executed.
"""

from typing import Any


def build_candidate_audit(
    *,
    payment_id: str,
    predicted_probabilities: dict[str, float],
    expected_values: dict[str, float] | None = None,
    winner: str | None,
    executed: bool = False,
    blocked_reason: str = "execution_mode_shadow",
) -> dict[str, Any]:
    expected_values = expected_values or {}
    candidates: dict[str, dict[str, float | None]] = {}
    for action, probability in predicted_probabilities.items():
        candidates[action] = {
            "probability": float(probability),
            "expected_value": (
                float(expected_values[action])
                if action in expected_values
                else None
            ),
        }
    return {
        "payment_id": payment_id,
        "candidates": candidates,
        "winner": winner,
        "executed": executed,
        "blocked_reason": blocked_reason,
    }
