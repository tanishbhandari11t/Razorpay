from __future__ import annotations

from pathlib import Path

import pandas as pd

from ml.src.controlled_execution_simulator import (
    simulate_controlled_execution,
)


def test_shadow_gate_blocks_every_simulated_action(tmp_path: Path) -> None:
    source = tmp_path / "shadow_cases.csv"
    output = tmp_path / "simulation.json"
    pd.DataFrame(
        [
            {
                "selected_action": "payment_link",
                "risk_checks_passed": True,
            },
            {
                "selected_action": "escalate_to_merchant",
                "risk_checks_passed": True,
            },
            {
                "selected_action": "no_action",
                "risk_checks_passed": True,
            },
        ]
    ).to_csv(source, index=False)

    report = simulate_controlled_execution(source, output)

    assert report["execution_mode"] == "shadow"
    assert report["execution_gate_outcomes"] == {"would_block": 3}
    assert report["policy_outcomes_if_controlled"] == {
        "would_escalate": 1,
        "would_execute": 1,
        "would_stop": 1,
    }
    assert report["provider_calls"] == 0
