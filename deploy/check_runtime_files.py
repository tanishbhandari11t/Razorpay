"""Warn (or fail) when files required to boot RecoverAI are missing."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = [
    ROOT / "ml" / "artifacts" / "recovery_model_v1.json",
    ROOT / "ml" / "artifacts" / "preprocessing_v1.joblib",
    ROOT / "ml" / "artifacts" / "calibration_v1.joblib",
    ROOT / "ml" / "artifacts" / "model_metadata.json",
    ROOT / "ml" / "config" / "execution_gate.yaml",
    ROOT / "ml" / "reports" / "phase8" / "phase8_report_manifest.json",
]


def main() -> int:
    missing = [path for path in REQUIRED if not path.is_file()]
    if not missing:
        print("RecoverAI runtime files are present.")
        return 0
    print("Missing RecoverAI runtime files:")
    for path in missing:
        print(f"  - {path.relative_to(ROOT).as_posix()}")
    print(
        "Copy the frozen model artifacts into ml/artifacts/ before deploying. "
        "They are gitignored by default."
    )
    if os.environ.get("REQUIRE_RUNTIME_ARTIFACTS") == "1":
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
