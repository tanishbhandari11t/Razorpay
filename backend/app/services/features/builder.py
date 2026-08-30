from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.services.features.customer_history import (
    OnlineFeatureUnavailable,
    get_payment_feature_context,
    load_prior_customer_history,
)
from app.services.features.temporal import (
    CurrentPaymentSnapshot,
    build_temporal_features,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
FEATURE_SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "feature_schema.yaml"
MODEL_MANIFEST_PATH = REPO_ROOT / "ml" / "config" / "dataset_manifest.yaml"


@lru_cache(maxsize=1)
def load_feature_schema() -> dict[str, Any]:
    schema = yaml.safe_load(FEATURE_SCHEMA_PATH.read_text(encoding="utf-8"))
    names = [feature["name"] for feature in schema["features"]]
    materialized = [
        feature
        for feature in schema["features"]
        if feature.get("online_materialized", True)
    ]
    if (
        len(names) != 53
        or len(materialized) != 52
        or len(names) != len(set(names))
    ):
        raise RuntimeError(
            "Feature schema must define 52 temporal features plus action"
        )
    manifest = yaml.safe_load(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    schema_model_inputs = [
        feature["name"]
        for feature in schema["features"]
        if feature["model_input"]
    ]
    manifest_inputs = [
        *manifest["numerical_features"],
        *manifest["categorical_features"],
    ]
    if set(schema_model_inputs) != set(manifest_inputs):
        raise RuntimeError("Feature schema and model manifest have diverged")
    return schema


def ordered_feature_names() -> list[str]:
    return [
        feature["name"]
        for feature in load_feature_schema()["features"]
        if feature.get("online_materialized", True)
    ]


def model_feature_names() -> list[str]:
    schema = load_feature_schema()
    manifest = yaml.safe_load(MODEL_MANIFEST_PATH.read_text(encoding="utf-8"))
    schema_inputs = {
        feature["name"]
        for feature in schema["features"]
        if feature["model_input"]
    }
    return [
        *[
            name
            for name in manifest["numerical_features"]
            if name in schema_inputs
        ],
        *[
            name
            for name in manifest["categorical_features"]
            if name in schema_inputs
        ],
    ]


def validate_feature_row(features: dict[str, Any]) -> None:
    expected = ordered_feature_names()
    if list(features) != expected:
        raise OnlineFeatureUnavailable(
            "Online feature order does not match canonical schema"
        )
    if any(value is None for value in features.values()):
        raise OnlineFeatureUnavailable("Online feature row contains null values")


def build_online_features(
    session: Session,
    recovery_case_id: str,
) -> dict[str, Any]:
    row = session.execute(
        select(RecoveryCase, Payment)
        .join(Payment, RecoveryCase.payment_id == Payment.id)
        .where(RecoveryCase.id == recovery_case_id)
    ).one_or_none()
    if row is None:
        raise OnlineFeatureUnavailable("Recovery case not found")
    recovery_case, payment = row
    if payment.customer_id is None or recovery_case.customer_id is None:
        raise OnlineFeatureUnavailable(
            "Recovery case has no persistent customer identity"
        )
    context = get_payment_feature_context(session, payment.id)
    history = load_prior_customer_history(session, payment)
    current = CurrentPaymentSnapshot(
        transaction_id=payment.id,
        customer_id=payment.customer_id,
        prediction_time=payment.created_at,
        amount_inr=float(payment.amount) / 100,
        transaction_type=context.transaction_type,
        merchant_category=context.merchant_category,
        device_type=context.device_type,
        network_type=context.network_type,
        fraud_flag=context.fraud_flag,
        sender_age_group=context.sender_age_group,
        sender_state=context.sender_state,
        sender_bank=context.sender_bank,
    )
    unordered = build_temporal_features(current, history)
    features = {name: unordered[name] for name in ordered_feature_names()}
    validate_feature_row(features)
    return features
