from __future__ import annotations

import math
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import Session


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ml.model_loader import load_model_bundle
from app.database.connection import Base
from app.models.customer import Customer
from app.models.payment import Payment
from app.models.payment_feature_context import PaymentFeatureContext
from app.models.recovery_case import RecoveryCase
from app.services.failure_mapping import map_razorpay_failure_reason
from app.services.features.builder import (
    build_online_features,
    model_feature_names,
    ordered_feature_names,
)
from app.services.features.temporal import (
    CurrentPaymentSnapshot,
    TransactionSnapshot,
    build_temporal_features,
)
from ml.src.build_features import build_features


def _row(
    transaction_id: str,
    timestamp: datetime,
    *,
    status: str,
    amount: int,
    transaction_type: str = "P2M",
    merchant_category: str = "Utilities",
    device_type: str = "Android",
    network_type: str = "4G",
    fraud_flag: int = 0,
) -> dict:
    return {
        "transaction_id": transaction_id,
        "customer_id": "CUST-PARITY",
        "timestamp": timestamp,
        "transaction_type": transaction_type,
        "merchant_category": merchant_category,
        "amount_inr": amount,
        "transaction_status": status,
        "sender_age_group": "26-35",
        "receiver_age_group": "36-45",
        "sender_state": "Karnataka",
        "sender_bank": "HDFC",
        "receiver_bank": "SBI",
        "device_type": device_type,
        "network_type": network_type,
        "fraud_flag": fraud_flag,
        "hour_of_day": timestamp.hour,
        "day_of_week": timestamp.strftime("%A"),
        "is_weekend": int(timestamp.weekday() >= 5),
    }


def _online(rows: list[dict], current_id: str) -> dict:
    current = next(row for row in rows if row["transaction_id"] == current_id)
    history = [
        TransactionSnapshot(
            transaction_id=row["transaction_id"],
            customer_id=row["customer_id"],
            timestamp=row["timestamp"],
            status=row["transaction_status"],
            amount_inr=float(row["amount_inr"]),
            transaction_type=row["transaction_type"],
            merchant_category=row["merchant_category"],
            device_type=row["device_type"],
            network_type=row["network_type"],
            fraud_flag=row["fraud_flag"],
        )
        for row in rows
        if row["transaction_id"] != current_id
    ]
    snapshot = CurrentPaymentSnapshot(
        transaction_id=current["transaction_id"],
        customer_id=current["customer_id"],
        prediction_time=current["timestamp"],
        amount_inr=float(current["amount_inr"]),
        transaction_type=current["transaction_type"],
        merchant_category=current["merchant_category"],
        device_type=current["device_type"],
        network_type=current["network_type"],
        fraud_flag=current["fraud_flag"],
        sender_age_group=current["sender_age_group"],
        sender_state=current["sender_state"],
        sender_bank=current["sender_bank"],
    )
    result = build_temporal_features(snapshot, history)
    return {name: result[name] for name in ordered_feature_names()}


def _assert_parity(expected: dict, actual: dict) -> None:
    assert list(actual) == ordered_feature_names()
    for name in ordered_feature_names():
        offline = expected[name]
        online = actual[name]
        if name == "prediction_time":
            assert pd.Timestamp(offline).to_pydatetime() == online
        elif isinstance(offline, (int, float)):
            assert math.isclose(
                float(offline),
                float(online),
                rel_tol=1e-12,
                abs_tol=1e-12,
            ), name
        else:
            assert offline == online, name


def test_online_features_exactly_match_offline_phase3_semantics() -> None:
    now = datetime(2026, 8, 25, 14, 0, tzinfo=UTC)
    rows = [
        _row("old", now - timedelta(days=31), status="SUCCESS", amount=100),
        _row("boundary-30", now - timedelta(days=30), status="FAILED", amount=200),
        _row("recent-success", now - timedelta(days=8), status="SUCCESS", amount=300),
        _row("boundary-7", now - timedelta(days=7), status="FAILED", amount=400),
        _row(
            "recent-failure",
            now - timedelta(days=6, hours=23),
            status="FAILED",
            amount=500,
            device_type="iOS",
            network_type="WiFi",
            fraud_flag=1,
        ),
        _row("target", now, status="FAILED", amount=350),
        _row("same-time", now, status="SUCCESS", amount=999),
        _row("future", now + timedelta(seconds=1), status="SUCCESS", amount=888),
    ]
    offline = build_features(pd.DataFrame(rows))
    expected = offline.loc[offline["transaction_id"] == "target"].iloc[0].to_dict()
    _assert_parity(expected, _online(rows, "target"))


def test_no_history_and_same_timestamp_are_explicitly_excluded() -> None:
    now = datetime(2026, 8, 25, 14, 0, tzinfo=UTC)
    rows = [
        _row("target", now, status="FAILED", amount=2499),
        _row("same-time", now, status="SUCCESS", amount=100),
    ]
    offline = build_features(pd.DataFrame(rows))
    expected = offline.loc[offline["transaction_id"] == "target"].iloc[0].to_dict()
    actual = _online(rows, "target")
    _assert_parity(expected, actual)
    assert actual["has_prior_history"] == 0
    assert actual["previous_transaction_count"] == 0
    assert actual["days_since_previous_transaction"] == -1.0
    assert actual["customer_primary_device_before_failure"] == "UNKNOWN"


def test_feature_order_matches_frozen_preprocessor() -> None:
    bundle = load_model_bundle()
    assert tuple(model_feature_names()) == bundle.raw_feature_names
    assert len(bundle.raw_feature_names) == 49
    assert bundle.transformed_feature_count == 103


def test_database_builder_fetches_only_strictly_prior_customer_history() -> None:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    now = datetime(2026, 8, 25, 14, 0, tzinfo=UTC)
    with Session(engine) as session:
        customer = Customer(id="customer", name="Parity Customer")
        prior = Payment(
            id="prior",
            customer_id=customer.id,
            razorpay_payment_id="pay_prior",
            amount=10000,
            currency="INR",
            status="captured",
            created_at=now - timedelta(days=1),
        )
        same_time = Payment(
            id="same-time",
            customer_id=customer.id,
            razorpay_payment_id="pay_same",
            amount=20000,
            currency="INR",
            status="captured",
            created_at=now,
        )
        current = Payment(
            id="current",
            customer_id=customer.id,
            razorpay_payment_id="pay_current",
            amount=30000,
            currency="INR",
            status="failed",
            failure_reason="authentication_failed",
            created_at=now,
        )
        session.add_all([customer, prior, same_time, current])
        session.flush()
        for payment in (prior, same_time, current):
            session.add(
                PaymentFeatureContext(
                    payment_id=payment.id,
                    transaction_type="P2M",
                    merchant_category="Utilities",
                    device_type="Android",
                    network_type="4G",
                    sender_age_group="26-35",
                    sender_state="Karnataka",
                    sender_bank="HDFC",
                )
            )
        recovery_case = RecoveryCase(
            id="case",
            payment_id=current.id,
            customer_id=customer.id,
            status="at_risk",
            source_event_id="event",
            created_at=now,
        )
        session.add(recovery_case)
        session.flush()

        features = build_online_features(session, recovery_case.id)

    assert features["previous_transaction_count"] == 1
    assert features["previous_success_count"] == 1
    assert features["previous_avg_amount"] == 100


def test_razorpay_failure_reason_mapping_is_deterministic() -> None:
    assert (
        map_razorpay_failure_reason(
            "international_transaction_not_allowed"
        )
        == "merchant_configuration"
    )
    assert (
        map_razorpay_failure_reason("anything", fraud_flag=1)
        == "fraud_risk"
    )
