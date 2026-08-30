from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ml.src.record_recovery_outcome import load_real_outcome_schema


REPO_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = REPO_ROOT / "ml" / "data" / "processed"
REAL_OUTPUT_PATH = PROCESSED_DIR / "real_observed_outcomes.csv"
SYNTHETIC_OUTPUT_PATH = PROCESSED_DIR / "synthetic_outcome_rows.csv"
TRAINING_CANDIDATE_PATH = PROCESSED_DIR / "real_controlled_training_candidates.csv"

REQUIRED_COLUMNS = [
    "decision_id",
    "payment_id",
    "action",
    "outcome_state",
    "outcome_source",
    "data_source",
    "attempted",
    "payment_recovered",
    "recovered_amount_minor",
    "failure_timestamp",
    "payment_status_after_24h",
    "payment_status_after_48h",
    "time_to_recovery_seconds",
]


class ProvenanceError(ValueError):
    pass


def split_outcomes_by_provenance(
    frame: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    schema = load_real_outcome_schema()
    dataset_config = schema["observation"]["evaluation_dataset"]
    if "data_source" not in frame.columns:
        raise ProvenanceError(
            "Outcome dataset missing data_source; refusing unlabeled mix"
        )
    if dataset_config["allow_unlabeled_mix"]:
        raise ProvenanceError("Unlabeled mixes are not allowed in Phase 14")
    allowed = set(schema["data_sources"])
    unknown = set(frame["data_source"].dropna().unique()) - allowed
    if unknown:
        raise ProvenanceError(
            f"Unknown outcome data_source values: {sorted(unknown)}"
        )
    real_sources = set(dataset_config["real_sources"])
    training_source = dataset_config["training_candidate_source"]
    attempted = frame["attempted"].fillna(False).astype(bool)
    return {
        "real": frame[frame["data_source"].isin(real_sources)].copy(),
        "synthetic": frame[frame["data_source"] == "synthetic"].copy(),
        "training_candidates": frame[
            (frame["data_source"] == training_source)
            & attempted
            & (frame["outcome_state"].isin(["recovered", "no_recovery_observed"]))
        ].copy(),
    }


def write_real_observed_outcomes(
    frame: pd.DataFrame,
    *,
    real_path: Path = REAL_OUTPUT_PATH,
    synthetic_path: Path = SYNTHETIC_OUTPUT_PATH,
    training_path: Path = TRAINING_CANDIDATE_PATH,
) -> dict[str, Any]:
    working = frame.copy()
    for column in REQUIRED_COLUMNS:
        if column not in working.columns:
            working[column] = None
    split = split_outcomes_by_provenance(working)
    real_path.parent.mkdir(parents=True, exist_ok=True)
    split["real"].to_csv(real_path, index=False)
    split["synthetic"].to_csv(synthetic_path, index=False)
    split["training_candidates"].to_csv(training_path, index=False)
    return {
        "real_rows": int(len(split["real"])),
        "synthetic_rows": int(len(split["synthetic"])),
        "training_candidate_rows": int(len(split["training_candidates"])),
        "mixed_without_provenance": False,
        "real_path": str(real_path.relative_to(REPO_ROOT)).replace("\\", "/"),
        "synthetic_path": str(
            synthetic_path.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
        "training_candidate_path": str(
            training_path.relative_to(REPO_ROOT)
        ).replace("\\", "/"),
    }
