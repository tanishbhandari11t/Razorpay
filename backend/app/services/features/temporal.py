from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from statistics import median
from typing import Any, Iterable


UNKNOWN = "UNKNOWN"
NO_HISTORY_DAYS = -1.0


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


@dataclass(frozen=True)
class TransactionSnapshot:
    transaction_id: str
    customer_id: str
    timestamp: datetime
    status: str
    amount_inr: float
    transaction_type: str
    merchant_category: str
    device_type: str
    network_type: str
    fraud_flag: int

    @property
    def hour_of_day(self) -> int:
        return _utc(self.timestamp).hour

    @property
    def day_of_week(self) -> str:
        return _utc(self.timestamp).strftime("%A")

    @property
    def is_weekend(self) -> int:
        return int(_utc(self.timestamp).weekday() >= 5)


@dataclass(frozen=True)
class CurrentPaymentSnapshot:
    transaction_id: str
    customer_id: str
    prediction_time: datetime
    amount_inr: float
    transaction_type: str
    merchant_category: str
    device_type: str
    network_type: str
    fraud_flag: int
    sender_age_group: str
    sender_state: str
    sender_bank: str


def _mode(values: Iterable[str]) -> str:
    counter = Counter(values)
    if not counter:
        return UNKNOWN
    maximum = max(counter.values())
    return sorted(key for key, count in counter.items() if count == maximum)[0]


def _days_since(
    prediction_time: datetime,
    previous_time: datetime | None,
) -> float:
    if previous_time is None:
        return NO_HISTORY_DAYS
    return (_utc(prediction_time) - _utc(previous_time)).total_seconds() / 86_400


def build_temporal_features(
    current: CurrentPaymentSnapshot,
    transactions: Iterable[TransactionSnapshot],
) -> dict[str, Any]:
    prediction_time = _utc(current.prediction_time)
    history = sorted(
        (
            transaction
            for transaction in transactions
            if _utc(transaction.timestamp) < prediction_time
        ),
        key=lambda value: (_utc(value.timestamp), value.transaction_id),
    )
    invalid_statuses = {
        transaction.status for transaction in history
    } - {"SUCCESS", "FAILED"}
    if invalid_statuses:
        raise ValueError(f"Invalid historical statuses: {sorted(invalid_statuses)}")
    if any(transaction.customer_id != current.customer_id for transaction in history):
        raise ValueError("Customer history contains another customer")

    previous_count = len(history)
    successes = [value for value in history if value.status == "SUCCESS"]
    failures = [value for value in history if value.status == "FAILED"]
    amounts = [float(value.amount_inr) for value in history]
    cutoff_7d = prediction_time - timedelta(days=7)
    cutoff_30d = prediction_time - timedelta(days=30)
    recent_7d = [
        value
        for value in history
        if _utc(value.timestamp) > cutoff_7d
    ]
    recent_30d = [
        value
        for value in history
        if _utc(value.timestamp) > cutoff_30d
    ]
    failure_streak = 0
    for transaction in reversed(history):
        if transaction.status != "FAILED":
            break
        failure_streak += 1

    average_amount = sum(amounts) / previous_count if previous_count else 0.0
    transaction_types = Counter(
        value.transaction_type for value in history
    )
    merchant_categories = Counter(
        value.merchant_category for value in history
    )
    days_of_week = Counter(value.day_of_week for value in history)
    primary_device = _mode(value.device_type for value in history)
    primary_network = _mode(value.network_type for value in history)
    current_time = prediction_time
    current_hour = current_time.hour
    current_day = current_time.strftime("%A")
    current_is_weekend = int(current_time.weekday() >= 5)
    close_hour_count = sum(
        min(
            abs(value.hour_of_day - current_hour),
            24 - abs(value.hour_of_day - current_hour),
        )
        <= 2
        for value in history
    )

    return {
        "transaction_id": current.transaction_id,
        "customer_id": current.customer_id,
        "prediction_time": prediction_time,
        "amount_inr": int(current.amount_inr),
        "transaction_type": current.transaction_type,
        "merchant_category": current.merchant_category,
        "device_type": current.device_type,
        "network_type": current.network_type,
        "fraud_flag": int(current.fraud_flag),
        "hour_of_day": current_hour,
        "day_of_week": current_day,
        "is_weekend": current_is_weekend,
        "sender_age_group": current.sender_age_group,
        "sender_state": current.sender_state,
        "sender_bank": current.sender_bank,
        "has_prior_history": int(previous_count > 0),
        "has_previous_success": int(bool(successes)),
        "has_previous_failure": int(bool(failures)),
        "previous_transaction_count": previous_count,
        "previous_success_count": len(successes),
        "previous_failure_count": len(failures),
        "historical_success_rate": (
            len(successes) / previous_count if previous_count else 0.0
        ),
        "historical_failure_rate": (
            len(failures) / previous_count if previous_count else 0.0
        ),
        "previous_failure_streak": failure_streak,
        "transactions_last_7d": len(recent_7d),
        "successes_last_7d": sum(
            value.status == "SUCCESS" for value in recent_7d
        ),
        "failures_last_7d": sum(
            value.status == "FAILED" for value in recent_7d
        ),
        "amount_last_7d": float(
            sum(value.amount_inr for value in recent_7d)
        ),
        "transactions_last_30d": len(recent_30d),
        "successes_last_30d": sum(
            value.status == "SUCCESS" for value in recent_30d
        ),
        "failures_last_30d": sum(
            value.status == "FAILED" for value in recent_30d
        ),
        "amount_last_30d": float(
            sum(value.amount_inr for value in recent_30d)
        ),
        "days_since_previous_transaction": _days_since(
            prediction_time,
            history[-1].timestamp if history else None,
        ),
        "days_since_previous_success": _days_since(
            prediction_time,
            successes[-1].timestamp if successes else None,
        ),
        "days_since_previous_failure": _days_since(
            prediction_time,
            failures[-1].timestamp if failures else None,
        ),
        "previous_avg_amount": average_amount,
        "previous_median_amount": float(median(amounts)) if amounts else 0.0,
        "previous_max_amount": max(amounts) if amounts else 0.0,
        "amount_vs_previous_avg": (
            float(current.amount_inr) / average_amount
            if average_amount > 0
            else 0.0
        ),
        "current_amount_percentile": (
            sum(amount <= float(current.amount_inr) for amount in amounts)
            / previous_count
            if previous_count
            else 0.0
        ),
        "same_transaction_type_previous_count": transaction_types[
            current.transaction_type
        ],
        "same_merchant_category_previous_count": merchant_categories[
            current.merchant_category
        ],
        "same_merchant_category_previous_rate": (
            merchant_categories[current.merchant_category] / previous_count
            if previous_count
            else 0.0
        ),
        "customer_primary_device_before_failure": primary_device,
        "device_matches_primary": int(
            previous_count > 0 and current.device_type == primary_device
        ),
        "customer_primary_network_before_failure": primary_network,
        "network_matches_primary": int(
            previous_count > 0 and current.network_type == primary_network
        ),
        "previous_fraud_count": sum(value.fraud_flag for value in history),
        "historical_fraud_rate": (
            sum(value.fraud_flag for value in history) / previous_count
            if previous_count
            else 0.0
        ),
        "historical_weekend_ratio": (
            sum(value.is_weekend for value in history) / previous_count
            if previous_count
            else 0.0
        ),
        "same_day_of_week_previous_rate": (
            days_of_week[current_day] / previous_count
            if previous_count
            else 0.0
        ),
        "usual_hour_previous_rate": (
            close_hour_count / previous_count if previous_count else 0.0
        ),
    }
