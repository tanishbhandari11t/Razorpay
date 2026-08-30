from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ml.src.razorpay_failure_taxonomy import (
    classify_razorpay_failure,
    legacy_failure_class,
    normalize_razorpay_reason,
)


def normalize_razorpay_failure_reason(reason: str | None) -> str:
    return normalize_razorpay_reason(reason)


def map_razorpay_failure_reason(
    reason: str | None,
    *,
    fraud_flag: int = 0,
) -> str:
    diagnosis = classify_razorpay_failure(
        reason,
        fraud_flag=fraud_flag,
    )
    return legacy_failure_class(diagnosis)
