from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.connection import get_session, initialize_database
from app.main import app
from app.ml.model_loader_v2_online import load_v2_online_model_bundle
from app.models.agent_decision import AgentDecision
from app.models.intervention import Intervention
from app.models.recovery_job import RecoveryJob


def _configure(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{(tmp_path / 'v2-shadow.db').as_posix()}",
    )
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_v2")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "v2-key-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "v2-webhook-secret")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("EXECUTION_MODE", "shadow")
    initialize_database()
    return "v2-webhook-secret"


def _headers(body: bytes, secret: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": hmac.new(
            secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest(),
        "X-Razorpay-Event-Id": "evt_v2_online_shadow",
    }


def test_v2_online_bundle_is_isolated() -> None:
    bundle = load_v2_online_model_bundle()
    assert bundle.model_version == "recovery_model_v2_online"
    assert bundle.policy_version == "recovery_policy_v3_v2_online"
    assert len(bundle.raw_feature_names) == 33
    assert bundle.transformed_feature_count == 43


def test_manual_v2_online_shadow_is_idempotent_and_nonexecuting(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch)
    payload = {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_v2_online_shadow",
                    "customer_id": "cust_v2_online_shadow",
                    "email": "v2@example.test",
                    "contact": "+910000000001",
                    "amount": 249900,
                    "currency": "INR",
                    "method": "netbanking",
                    "bank": "BARB_R",
                    "status": "failed",
                    "error_reason": "payment_failed",
                    "created_at": int(datetime.now(UTC).timestamp()),
                }
            }
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    with TestClient(app) as client:
        webhook = client.post(
            "/webhooks/razorpay",
            content=body,
            headers=_headers(body, secret),
        )
        case_id = webhook.json()["recovery_case_id"]
        status = client.get("/api/recovery/shadow/v2-online/status")
        first = client.post(
            f"/api/recovery/shadow/v2-online/{case_id}/evaluate"
        )
        second = client.post(
            f"/api/recovery/shadow/v2-online/{case_id}/evaluate"
        )
        comparison = client.get(
            f"/api/recovery/cases/{case_id}/decisions/compare"
        )

    assert webhook.status_code == 200
    assert webhook.json()["v2_online_shadow_job"] is None
    assert status.status_code == 200
    assert status.json()["enabled"] is False
    assert status.json()["provider_actions_enabled"] is False
    assert first.status_code == 200
    assert first.json()["model_version"] == "recovery_model_v2_online"
    assert first.json()["selected_action"] == "escalate_to_merchant"
    assert first.json()["execution_mode"] == "shadow"
    assert first.json()["executed"] is False
    assert second.status_code == 200
    assert second.json()["idempotent_replay"] is True
    assert comparison.status_code == 200
    assert comparison.json()["paired"] is True
    assert comparison.json()["action_agreement"] is True

    with get_session() as session:
        decisions = session.scalars(select(AgentDecision)).all()
        interventions = session.scalars(select(Intervention)).all()
        jobs = session.scalars(select(RecoveryJob)).all()
    assert len(decisions) == 2
    assert {decision.model_version for decision in decisions} == {
        "recovery_model_v1",
        "recovery_model_v2_online",
    }
    assert all(decision.execution_mode == "shadow" for decision in decisions)
    assert len(interventions) == 0
    assert len(jobs) == 1
    assert jobs[0].task_name == "shadow_inference"
