from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.connection import get_session, initialize_database
from app.main import app
from app.models.agent_decision import AgentDecision
from app.models.recovery_job import RecoveryJob
from app.workers.recovery_tasks import process_recovery_case
from app.workers.celery_app import celery_app


def _configure(tmp_path, monkeypatch) -> str:
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{(tmp_path / 'e2e.db').as_posix()}",
    )
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_e2e")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "e2e-key-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "e2e-webhook-secret")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("EXECUTION_MODE", "shadow")
    initialize_database()
    return "e2e-webhook-secret"


def _headers(body: bytes, secret: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": hmac.new(
            secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest(),
        "X-Razorpay-Event-Id": "evt_e2e_shadow",
    }


def test_e2e_webhook_queue_worker_inference_and_execution_gate(
    tmp_path,
    monkeypatch,
) -> None:
    assert celery_app.conf.task_acks_late is True
    assert celery_app.conf.task_reject_on_worker_lost is True
    assert celery_app.conf.worker_prefetch_multiplier == 1
    secret = _configure(tmp_path, monkeypatch)
    payload = {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_e2e_shadow",
                    "customer_id": "cust_e2e_shadow",
                    "email": "shadow@example.test",
                    "amount": 249900,
                    "currency": "INR",
                    "method": "upi",
                    "status": "failed",
                    "error_reason": "international_transaction_not_allowed",
                    "created_at": int(datetime.now(UTC).timestamp()),
                    "notes": {
                        "transaction_type": "P2M",
                        "merchant_category": "Utilities",
                        "device_type": "Android",
                        "network_type": "4G",
                        "sender_age_group": "26-35",
                        "sender_state": "Karnataka",
                        "sender_bank": "HDFC",
                    },
                }
            }
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode()

    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=body,
            headers=_headers(body, secret),
        )
        gate = client.get("/api/recovery/shadow/gate")

    assert response.status_code == 200
    assert response.json()["recovery_job"]["status"] == "succeeded"
    assert response.json()["recovery_job"]["published"] is True
    assert gate.status_code == 200
    assert gate.json()["status"] == "blocked"
    assert gate.json()["execution_mode"] == "shadow"
    assert gate.json()["checks"]["minimum_real_cases"]["passed"] is False

    with get_session() as session:
        jobs = session.scalars(select(RecoveryJob)).all()
        decisions = session.scalars(select(AgentDecision)).all()
    assert len(jobs) == 1
    assert jobs[0].status == "succeeded"
    assert jobs[0].attempts == 1
    assert len(decisions) == 1
    assert decisions[0].execution_mode == "shadow"
    execution_check = next(
        check
        for check in decisions[0].risk_checks
        if check["name"] == "execution_gate_blocked"
    )
    assert execution_check == {
        "name": "execution_gate_blocked",
        "passed": True,
        "reason": "execution_mode_shadow",
    }

    replay = process_recovery_case.apply(
        args=[jobs[0].recovery_case_id, jobs[0].id],
    ).get()
    assert replay["idempotent_replay"] is True
    with get_session() as session:
        assert len(session.scalars(select(AgentDecision)).all()) == 1
