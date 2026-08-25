from __future__ import annotations

import pandas as pd
import pytest

from ml.src.build_features import (
    NO_HISTORY_DAYS,
    REQUIRED_COLUMNS,
    build_features,
    validate_temporal_integrity,
)


def transaction(
    transaction_id: str,
    customer_id: str,
    timestamp: str,
    status: str,
    amount: int,
    *,
    merchant: str = "Grocery",
    device: str = "Android",
    network: str = "4G",
) -> dict[str, object]:
    parsed = pd.Timestamp(timestamp)
    return {
        "transaction_id": transaction_id,
        "customer_id": customer_id,
        "timestamp": timestamp,
        "transaction_type": "P2M",
        "merchant_category": merchant,
        "amount_inr": amount,
        "transaction_status": status,
        "sender_age_group": "26-35",
        "receiver_age_group": "26-35",
        "sender_state": "Karnataka",
        "sender_bank": "HDFC",
        "receiver_bank": "SBI",
        "device_type": device,
        "network_type": network,
        "fraud_flag": 0,
        "hour_of_day": parsed.hour,
        "day_of_week": parsed.day_name(),
        "is_weekend": int(parsed.dayofweek >= 5),
    }


def fixture_transactions() -> pd.DataFrame:
    rows = [
        transaction("T001", "C001", "2024-01-01 10:00:00", "SUCCESS", 100),
        transaction("T002", "C001", "2024-01-05 10:00:00", "SUCCESS", 200),
        transaction("T003", "C001", "2024-01-10 10:00:00", "FAILED", 300),
        transaction("T004", "C001", "2024-01-10 10:00:00", "SUCCESS", 400),
        transaction("T005", "C001", "2024-01-20 10:00:00", "FAILED", 500),
        transaction("T006", "C002", "2024-01-02 08:00:00", "FAILED", 150),
    ]
    return pd.DataFrame(rows, columns=REQUIRED_COLUMNS)


def test_current_and_same_timestamp_transactions_are_excluded() -> None:
    source = fixture_transactions()
    features = build_features(source).set_index("transaction_id")

    first_failure = features.loc["T003"]
    assert first_failure["previous_transaction_count"] == 2
    assert first_failure["previous_success_count"] == 2
    assert first_failure["previous_failure_count"] == 0
    assert first_failure["previous_avg_amount"] == pytest.approx(150)
    assert first_failure["previous_median_amount"] == pytest.approx(150)
    assert first_failure["amount_vs_previous_avg"] == pytest.approx(2)

    later_failure = features.loc["T005"]
    assert later_failure["previous_transaction_count"] == 4
    assert later_failure["previous_success_count"] == 3
    assert later_failure["previous_failure_count"] == 1
    assert later_failure["previous_failure_streak"] == 0


def test_strict_recent_windows_and_recency() -> None:
    features = build_features(fixture_transactions()).set_index("transaction_id")
    later_failure = features.loc["T005"]

    assert later_failure["transactions_last_7d"] == 0
    assert later_failure["transactions_last_30d"] == 4
    assert later_failure["successes_last_30d"] == 3
    assert later_failure["failures_last_30d"] == 1
    assert later_failure["days_since_previous_transaction"] == pytest.approx(10)
    assert later_failure["days_since_previous_success"] == pytest.approx(10)
    assert later_failure["days_since_previous_failure"] == pytest.approx(10)


def test_first_transaction_failure_uses_explicit_sentinels() -> None:
    features = build_features(fixture_transactions()).set_index("transaction_id")
    first_transaction = features.loc["T006"]

    assert first_transaction["has_prior_history"] == 0
    assert first_transaction["previous_transaction_count"] == 0
    assert first_transaction["historical_success_rate"] == 0
    assert first_transaction["previous_avg_amount"] == 0
    assert first_transaction["days_since_previous_transaction"] == NO_HISTORY_DAYS
    assert first_transaction["customer_primary_device_before_failure"] == "UNKNOWN"


def test_future_transactions_do_not_change_existing_features() -> None:
    source = fixture_transactions()
    baseline = build_features(source).set_index("transaction_id").sort_index()
    future = transaction(
        "T999",
        "C001",
        "2024-02-20 10:00:00",
        "SUCCESS",
        50_000,
        merchant="Travel",
        device="iOS",
        network="5G",
    )
    extended = build_features(
        pd.concat([source, pd.DataFrame([future])], ignore_index=True)
    ).set_index("transaction_id").sort_index()

    pd.testing.assert_frame_equal(baseline, extended.loc[baseline.index])


def test_integrity_validator_accepts_valid_features() -> None:
    source = fixture_transactions()
    features = build_features(source)
    summary = validate_temporal_integrity(source, features)

    assert summary["feature_rows"] == 3
    assert summary["temporal_leakage_violations"] == 0
    assert summary["rolling_window_violations"] == 0


def test_missing_required_column_is_rejected() -> None:
    source = fixture_transactions().drop(columns=["customer_id"])
    with pytest.raises(ValueError, match="Missing required columns"):
        build_features(source)
