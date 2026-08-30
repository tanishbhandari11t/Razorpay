from __future__ import annotations

"""
Phase 15 V3 challenger entrypoint.

Refuses to train unless Phase 15 is explicitly authorized by the evidence gate.
V1 and V2-online remain frozen.
"""

import json
from pathlib import Path
from typing import Any

import yaml

from ml.src.evidence_coverage import phase15_authorization
from ml.src.observe_recovery import load_outcome_observation_config


REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "ml" / "config" / "recovery_model_v3_challenger.yaml"
REPORT_DIR = REPO_ROOT / "ml" / "reports" / "phase15"
READINESS_PATH = REPORT_DIR / "phase15_readiness.json"


class ChallengerBlockedError(RuntimeError):
    pass


def load_v3_config() -> dict[str, Any]:
    return yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))


def assert_training_authorized(
    evidence_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_v3_config()
    observation = load_outcome_observation_config()
    authorization = phase15_authorization(evidence_records or [])
    if observation["safety"]["phase15_authorized"]:
        raise ChallengerBlockedError(
            "outcome_observation.yaml must keep phase15_authorized=false "
            "until an explicit later gate flip"
        )
    if config["readiness"]["phase15_authorized"]:
        raise ChallengerBlockedError(
            "V3 challenger config cannot self-authorize Phase 15"
        )
    if config["training"]["require_phase15_authorized"] and not authorization[
        "authorized"
    ]:
        raise ChallengerBlockedError(
            "Phase 15 training blocked: evidence coverage gate unauthorized"
        )
    return authorization


def write_phase15_readiness(
    evidence_records: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    config = load_v3_config()
    authorization = phase15_authorization(evidence_records or [])
    blocked_reason = None
    try:
        assert_training_authorized(evidence_records)
    except ChallengerBlockedError as exc:
        blocked_reason = str(exc)
    readiness = {
        "version": 1,
        "phase": 15,
        "model_version": config["identity"]["model_version"],
        "model_ready": False,
        "phase15_authorized": False,
        "controlled_execution_authorized": False,
        "candidate_ready": authorization["candidate_ready"],
        "authorization": authorization,
        "blocked_reason": blocked_reason
        or "Insufficient real attributed evidence for challenger training",
        "frozen_parents": config["identity"]["parent_models"],
        "next_step": (
            "Continue Phase 14C/14D evidence collection. Do not train V3."
        ),
    }
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    READINESS_PATH.write_text(
        json.dumps(readiness, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return readiness


def main() -> None:
    readiness = write_phase15_readiness([])
    print(json.dumps(readiness, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
