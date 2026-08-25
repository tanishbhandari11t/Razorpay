from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "transactions_with_customers.csv"
)
DEFAULT_OUTPUT_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "failed_payment_features.csv"
)
DEFAULT_SUMMARY_PATH = (
    REPO_ROOT
    / "ml"
    / "data"
    / "processed"
    / "temporal_feature_summary.json"
)

REQUIRED_COLUMNS = [
    "transaction_id",
    "customer_id",
    "timestamp",
    "transaction_type",
    "merchant_category",
    "amount_inr",
    "transaction_status",
    "sender_age_group",
    "receiver_age_group",
    "sender_state",
    "sender_bank",
    "receiver_bank",
    "device_type",
    "network_type",
    "fraud_flag",
    "hour_of_day",
    "day_of_week",
    "is_weekend",
]

VALID_STATUSES = {"SUCCESS", "FAILED"}
UNKNOWN_CATEGORY = "UNKNOWN"
NO_HISTORY_DAYS = -1.0


@dataclass(frozen=True)
class HistoricalTransaction:
    timestamp: pd.Timestamp
    status: str
    amount: float
    transaction_type: str
    merchant_category: str
    device_type: str
    network_type: str
    fraud_flag: int
    hour_of_day: int
    day_of_week: str
    is_weekend: int


def _validate_input(dataframe: pd.DataFrame) -> None:
    missing = set(REQUIRED_COLUMNS) - set(dataframe.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    if dataframe["transaction_id"].duplicated().any():
        raise ValueError("Duplicate transaction IDs found")
    if dataframe["customer_id"].isna().any():
        raise ValueError("Transactions without customer IDs found")
    invalid_statuses = set(dataframe["transaction_status"].unique()) - VALID_STATUSES
    if invalid_statuses:
        raise ValueError(f"Invalid transaction statuses: {sorted(invalid_statuses)}")
    if dataframe.isna().any().any():
        raise ValueError("Missing values found in customer transactions")


def _mode(counter: Counter[str]) -> str:
    if not counter:
        return UNKNOWN_CATEGORY
    maximum = max(counter.values())
    return sorted(key for key, count in counter.items() if count == maximum)[0]


def _days_since(
    prediction_time: pd.Timestamp,
    previous_time: pd.Timestamp | None,
) -> float:
    if previous_time is None:
        return NO_HISTORY_DAYS
    return (prediction_time - previous_time).total_seconds() / 86_400


def _window_summary(
    transactions: Iterable[HistoricalTransaction],
) -> dict[str, float | int]:
    values = list(transactions)
    return {
        "transactions": len(values),
        "successes": sum(value.status == "SUCCESS" for value in values),
        "failures": sum(value.status == "FAILED" for value in values),
        "amount": float(sum(value.amount for value in values)),
    }


def _historical_features(
    row: Mapping[str, Any],
    history: list[HistoricalTransaction],
    recent_7d: deque[HistoricalTransaction],
    recent_30d: deque[HistoricalTransaction],
    success_count: int,
    failure_count: int,
    failure_streak: int,
    amount_sum: float,
    amount_values: list[float],
    transaction_types: Counter[str],
    merchant_categories: Counter[str],
    devices: Counter[str],
    networks: Counter[str],
    fraud_count: int,
    hours: list[int],
    days_of_week: Counter[str],
    weekend_count: int,
    last_transaction_at: pd.Timestamp | None,
    last_success_at: pd.Timestamp | None,
    last_failure_at: pd.Timestamp | None,
) -> dict[str, Any]:
    prediction_time = pd.Timestamp(row["timestamp"])
    previous_count = len(history)
    has_history = int(previous_count > 0)
    has_success = int(success_count > 0)
    has_failure = int(failure_count > 0)
    average_amount = amount_sum / previous_count if previous_count else 0.0
    median_amount = float(np.median(amount_values)) if amount_values else 0.0
    maximum_amount = max(amount_values) if amount_values else 0.0
    current_amount = float(row["amount_inr"])
    amount_percentile = (
        sum(amount <= current_amount for amount in amount_values) / previous_count
        if previous_count
        else 0.0
    )
    window_7d = _window_summary(recent_7d)
    window_30d = _window_summary(recent_30d)
    primary_device = _mode(devices)
    primary_network = _mode(networks)
    current_hour = int(row["hour_of_day"])
    close_hour_count = sum(
        min(abs(hour - current_hour), 24 - abs(hour - current_hour)) <= 2
        for hour in hours
    )

    return {
        "transaction_id": str(row["transaction_id"]),
        "customer_id": str(row["customer_id"]),
        "prediction_time": prediction_time,
        "amount_inr": int(row["amount_inr"]),
        "transaction_type": str(row["transaction_type"]),
        "merchant_category": str(row["merchant_category"]),
        "device_type": str(row["device_type"]),
        "network_type": str(row["network_type"]),
        "fraud_flag": int(row["fraud_flag"]),
        "hour_of_day": current_hour,
        "day_of_week": str(row["day_of_week"]),
        "is_weekend": int(row["is_weekend"]),
        "sender_age_group": str(row["sender_age_group"]),
        "sender_state": str(row["sender_state"]),
        "sender_bank": str(row["sender_bank"]),
        "has_prior_history": has_history,
        "has_previous_success": has_success,
        "has_previous_failure": has_failure,
        "previous_transaction_count": previous_count,
        "previous_success_count": success_count,
        "previous_failure_count": failure_count,
        "historical_success_rate": (
            success_count / previous_count if previous_count else 0.0
        ),
        "historical_failure_rate": (
            failure_count / previous_count if previous_count else 0.0
        ),
        "previous_failure_streak": failure_streak,
        "transactions_last_7d": window_7d["transactions"],
        "successes_last_7d": window_7d["successes"],
        "failures_last_7d": window_7d["failures"],
        "amount_last_7d": window_7d["amount"],
        "transactions_last_30d": window_30d["transactions"],
        "successes_last_30d": window_30d["successes"],
        "failures_last_30d": window_30d["failures"],
        "amount_last_30d": window_30d["amount"],
        "days_since_previous_transaction": _days_since(
            prediction_time,
            last_transaction_at,
        ),
        "days_since_previous_success": _days_since(
            prediction_time,
            last_success_at,
        ),
        "days_since_previous_failure": _days_since(
            prediction_time,
            last_failure_at,
        ),
        "previous_avg_amount": average_amount,
        "previous_median_amount": median_amount,
        "previous_max_amount": maximum_amount,
        "amount_vs_previous_avg": (
            current_amount / average_amount if average_amount > 0 else 0.0
        ),
        "current_amount_percentile": amount_percentile,
        "same_transaction_type_previous_count": transaction_types[
            str(row["transaction_type"])
        ],
        "same_merchant_category_previous_count": merchant_categories[
            str(row["merchant_category"])
        ],
        "same_merchant_category_previous_rate": (
            merchant_categories[str(row["merchant_category"])] / previous_count
            if previous_count
            else 0.0
        ),
        "customer_primary_device_before_failure": primary_device,
        "device_matches_primary": int(
            has_history and str(row["device_type"]) == primary_device
        ),
        "customer_primary_network_before_failure": primary_network,
        "network_matches_primary": int(
            has_history and str(row["network_type"]) == primary_network
        ),
        "previous_fraud_count": fraud_count,
        "historical_fraud_rate": fraud_count / previous_count if previous_count else 0.0,
        "historical_weekend_ratio": (
            weekend_count / previous_count if previous_count else 0.0
        ),
        "same_day_of_week_previous_rate": (
            days_of_week[str(row["day_of_week"])] / previous_count
            if previous_count
            else 0.0
        ),
        "usual_hour_previous_rate": (
            close_hour_count / previous_count if previous_count else 0.0
        ),
    }


def _build_customer_features(customer: pd.DataFrame) -> list[dict[str, Any]]:
    history: list[HistoricalTransaction] = []
    recent_7d: deque[HistoricalTransaction] = deque()
    recent_30d: deque[HistoricalTransaction] = deque()
    success_count = 0
    failure_count = 0
    failure_streak = 0
    amount_sum = 0.0
    amount_values: list[float] = []
    transaction_types: Counter[str] = Counter()
    merchant_categories: Counter[str] = Counter()
    devices: Counter[str] = Counter()
    networks: Counter[str] = Counter()
    fraud_count = 0
    hours: list[int] = []
    days_of_week: Counter[str] = Counter()
    weekend_count = 0
    last_transaction_at: pd.Timestamp | None = None
    last_success_at: pd.Timestamp | None = None
    last_failure_at: pd.Timestamp | None = None
    feature_rows: list[dict[str, Any]] = []

    records = customer.to_dict(orient="records")
    position = 0
    while position < len(records):
        prediction_time = pd.Timestamp(records[position]["timestamp"])
        batch_end = position + 1
        while (
            batch_end < len(records)
            and pd.Timestamp(records[batch_end]["timestamp"]) == prediction_time
        ):
            batch_end += 1

        cutoff_7d = prediction_time - pd.Timedelta(days=7)
        cutoff_30d = prediction_time - pd.Timedelta(days=30)
        while recent_7d and recent_7d[0].timestamp <= cutoff_7d:
            recent_7d.popleft()
        while recent_30d and recent_30d[0].timestamp <= cutoff_30d:
            recent_30d.popleft()

        for row in records[position:batch_end]:
            if row["transaction_status"] == "FAILED":
                feature_rows.append(
                    _historical_features(
                        row,
                        history,
                        recent_7d,
                        recent_30d,
                        success_count,
                        failure_count,
                        failure_streak,
                        amount_sum,
                        amount_values,
                        transaction_types,
                        merchant_categories,
                        devices,
                        networks,
                        fraud_count,
                        hours,
                        days_of_week,
                        weekend_count,
                        last_transaction_at,
                        last_success_at,
                        last_failure_at,
                    )
                )

        # The complete same-timestamp batch becomes history only after scoring.
        batch = sorted(
            records[position:batch_end],
            key=lambda value: str(value["transaction_id"]),
        )
        for row in batch:
            transaction = HistoricalTransaction(
                timestamp=prediction_time,
                status=str(row["transaction_status"]),
                amount=float(row["amount_inr"]),
                transaction_type=str(row["transaction_type"]),
                merchant_category=str(row["merchant_category"]),
                device_type=str(row["device_type"]),
                network_type=str(row["network_type"]),
                fraud_flag=int(row["fraud_flag"]),
                hour_of_day=int(row["hour_of_day"]),
                day_of_week=str(row["day_of_week"]),
                is_weekend=int(row["is_weekend"]),
            )
            history.append(transaction)
            recent_7d.append(transaction)
            recent_30d.append(transaction)
            amount_sum += transaction.amount
            amount_values.append(transaction.amount)
            transaction_types[transaction.transaction_type] += 1
            merchant_categories[transaction.merchant_category] += 1
            devices[transaction.device_type] += 1
            networks[transaction.network_type] += 1
            fraud_count += transaction.fraud_flag
            hours.append(transaction.hour_of_day)
            days_of_week[transaction.day_of_week] += 1
            weekend_count += transaction.is_weekend
            last_transaction_at = prediction_time

            if transaction.status == "SUCCESS":
                success_count += 1
                failure_streak = 0
                last_success_at = prediction_time
            else:
                failure_count += 1
                failure_streak += 1
                last_failure_at = prediction_time
        position = batch_end

    return feature_rows


def build_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    _validate_input(dataframe)
    transactions = dataframe.copy()
    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"],
        errors="raise",
    )
    transactions = transactions.sort_values(
        ["customer_id", "timestamp", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    for _, customer in transactions.groupby("customer_id", sort=True, observed=True):
        rows.extend(_build_customer_features(customer))

    features = pd.DataFrame(rows)
    features = features.sort_values(
        ["customer_id", "prediction_time", "transaction_id"],
        kind="mergesort",
    ).reset_index(drop=True)
    if features.isna().any().any():
        raise ValueError("Temporal feature output contains missing values")
    return features


def validate_temporal_integrity(
    source: pd.DataFrame,
    features: pd.DataFrame,
) -> dict[str, Any]:
    source = source.copy()
    source["timestamp"] = pd.to_datetime(source["timestamp"], errors="raise")
    failed = source.loc[source["transaction_status"].eq("FAILED")]
    if len(features) != len(failed):
        raise ValueError(
            f"Expected {len(failed)} failed-payment rows, got {len(features)}"
        )
    if set(features["transaction_id"]) != set(failed["transaction_id"]):
        raise ValueError("Feature rows do not match the failed-payment population")
    if features["transaction_id"].duplicated().any():
        raise ValueError("Duplicate feature rows found")
    if not features["previous_transaction_count"].eq(
        features["previous_success_count"] + features["previous_failure_count"]
    ).all():
        raise ValueError("Historical count components are inconsistent")

    feature_lookup = features.set_index("transaction_id")
    leakage_violations = 0
    window_violations = 0
    for _, customer in source.groupby("customer_id", sort=True, observed=True):
        customer = customer.sort_values(
            ["timestamp", "transaction_id"],
            kind="mergesort",
        ).reset_index(drop=True)
        timestamps = pd.DatetimeIndex(customer["timestamp"])
        failed_rows = customer.loc[customer["transaction_status"].eq("FAILED")]
        for current in failed_rows.itertuples(index=False):
            prediction_time = pd.Timestamp(current.timestamp)
            previous_end = int(timestamps.searchsorted(prediction_time, side="left"))
            feature = feature_lookup.loc[current.transaction_id]
            if previous_end != int(feature["previous_transaction_count"]):
                leakage_violations += 1
            if previous_end and timestamps[previous_end - 1] >= prediction_time:
                leakage_violations += 1

            start_7d = int(
                timestamps.searchsorted(
                    prediction_time - pd.Timedelta(days=7),
                    side="right",
                )
            )
            start_30d = int(
                timestamps.searchsorted(
                    prediction_time - pd.Timedelta(days=30),
                    side="right",
                )
            )
            if previous_end - start_7d != int(feature["transactions_last_7d"]):
                window_violations += 1
            if previous_end - start_30d != int(feature["transactions_last_30d"]):
                window_violations += 1

    if leakage_violations:
        raise ValueError(f"Temporal leakage violations: {leakage_violations}")
    if window_violations:
        raise ValueError(f"Rolling-window violations: {window_violations}")

    return {
        "feature_rows": len(features),
        "failed_source_rows": len(failed),
        "customers_with_failed_payments": int(features["customer_id"].nunique()),
        "no_history_rows": int(features["has_prior_history"].eq(0).sum()),
        "temporal_leakage_violations": 0,
        "rolling_window_violations": 0,
        "minimum_previous_transactions": int(
            features["previous_transaction_count"].min()
        ),
        "median_previous_transactions": float(
            features["previous_transaction_count"].median()
        ),
        "maximum_previous_transactions": int(
            features["previous_transaction_count"].max()
        ),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run(
    input_path: Path,
    output_path: Path,
    summary_path: Path,
) -> dict[str, Any]:
    source = pd.read_csv(input_path, parse_dates=["timestamp"])
    features = build_features(source)
    summary = validate_temporal_integrity(source, features)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    features.to_csv(
        output_path,
        index=False,
        date_format="%Y-%m-%d %H:%M:%S",
        float_format="%.8f",
    )
    summary["input_sha256"] = _sha256(input_path)
    summary["features_sha256"] = _sha256(output_path)
    summary["history_rule"] = "history_timestamp_strictly_before_prediction_time"
    summary["no_history_numeric_sentinel"] = 0
    summary["no_history_recency_sentinel_days"] = NO_HISTORY_DAYS
    summary["no_history_category_sentinel"] = UNKNOWN_CATEGORY
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print("=== TEMPORAL FEATURE SUMMARY ===")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Saved features: {output_path}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build leakage-safe features for failed UPI payments."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_PATH)
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    run(arguments.input, arguments.output, arguments.summary_output)
