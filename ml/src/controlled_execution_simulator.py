from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPO_ROOT / "ml" / "reports" / "phase10" / "shadow_cases.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "reports"
    / "phase10"
    / "controlled_execution_simulation.json"
)
EXECUTION_GATE_PATH = REPO_ROOT / "ml" / "config" / "execution_gate.yaml"


def simulate_controlled_execution(
    input_path: Path = DEFAULT_INPUT_PATH,
    output_path: Path = DEFAULT_OUTPUT_PATH,
) -> dict[str, Any]:
    gate = yaml.safe_load(EXECUTION_GATE_PATH.read_text(encoding="utf-8"))
    mode = str(gate["execution"]["mode"])
    rows = pd.read_csv(input_path) if input_path.exists() else pd.DataFrame()
    outcomes: list[str] = []
    policy_outcomes: list[str] = []
    if not rows.empty:
        for row in rows.to_dict(orient="records"):
            action = str(row.get("selected_action") or "no_action")
            risk_passed = bool(row.get("risk_checks_passed", False))
            if action == "no_action":
                policy_outcome = "would_stop"
            elif action == "escalate_to_merchant":
                policy_outcome = "would_escalate"
            elif risk_passed:
                policy_outcome = "would_execute"
            else:
                policy_outcome = "would_block"
            policy_outcomes.append(policy_outcome)
            outcomes.append(
                policy_outcome
                if mode == "controlled"
                and gate["execution"]["provider_actions_enabled"]
                else "would_block"
            )
    report = {
        "version": 1,
        "execution_mode": mode,
        "provider_actions_enabled": bool(
            gate["execution"]["provider_actions_enabled"]
        ),
        "shadow_cases": len(rows),
        "policy_outcomes_if_controlled": dict(Counter(policy_outcomes)),
        "execution_gate_outcomes": dict(Counter(outcomes)),
        "provider_calls": 0,
        "status": "evaluated" if len(rows) else "awaiting_shadow_cases",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    print(
        json.dumps(
            simulate_controlled_execution(args.input, args.output),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
