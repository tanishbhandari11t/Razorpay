import hashlib
import hmac
import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.connection import get_session, initialize_database
from app.main import app
from app.models.agent_decision import AgentDecision
from app.models.intervention import Intervention
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.models.recovery_job import RecoveryJob
from app.models.webhook_event import WebhookEvent


def configure_test_database(tmp_path, monkeypatch) -> str:
    database_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_unit")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "unit-key-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "unit-webhook-secret")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("EXECUTION_MODE", "shadow")
    initialize_database()
    return "unit-webhook-secret"


def signed_headers(body: bytes, event_id: str, secret: str) -> dict[str, str]:
    signature = hmac.new(secret.encode(), body, hashlib.sha256).hexdigest()
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": signature,
        "X-Razorpay-Event-Id": event_id,
    }


def test_simulation_is_deterministic_and_bounded() -> None:
    with TestClient(app) as client:
        response = client.post("/api/simulations", json={"transactions": 500, "seed": 7})
        assert response.status_code == 200
        metrics = response.json()

        assert metrics["transactions"] == 500
        assert metrics["failed_transactions"] == 60
        assert metrics["revenue_at_risk"] > 0
        assert 0 <= metrics["recovery_rate"] <= 1
        assert metrics["policy_violations"] == 0
        assert metrics["duplicate_actions"] == 0

        cases = client.get("/api/cases?limit=100").json()
        assert len(cases) == 60
        assert all(case["interventions"] <= 3 for case in cases)
        assert all(case["audit"] for case in cases)


def test_unknown_case_returns_404() -> None:
    with TestClient(app) as client:
        response = client.get("/api/cases/REC-NOT-FOUND")
        assert response.status_code == 404


def test_cors_allows_local_dashboard() -> None:
    with TestClient(app) as client:
        response = client.get(
            "/api/policy",
            headers={"Origin": "http://localhost:5173"},
        )
        assert response.status_code == 200
        assert (
            response.headers.get("access-control-allow-origin")
            == "http://localhost:5173"
        )


def test_razorpay_status_never_exposes_secret() -> None:
    with TestClient(app) as client:
        response = client.get("/api/razorpay/status")
        assert response.status_code == 200
        assert "secret" not in response.text.lower()


def test_unsigned_webhook_is_rejected() -> None:
    with TestClient(app) as client:
        response = client.post(
            "/api/webhooks/razorpay",
            json={"event": "payment.failed", "payload": {}},
        )
        assert response.status_code == 400


def test_invalid_signature_is_rejected(tmp_path, monkeypatch) -> None:
    configure_test_database(tmp_path, monkeypatch)
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=b'{"event":"payment.failed","payload":{}}',
            headers={
                "Content-Type": "application/json",
                "X-Razorpay-Signature": "invalid",
                "X-Razorpay-Event-Id": "evt_invalid",
            },
        )
        assert response.status_code == 401


def test_payment_failed_creates_one_recovery_case(tmp_path, monkeypatch) -> None:
    secret = configure_test_database(tmp_path, monkeypatch)
    payload = {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_test_failed",
                    "order_id": "order_test",
                    "customer_id": "cust_test",
                    "email": "amit@example.test",
                    "contact": "+910000000000",
                    "amount": 249900,
                    "currency": "INR",
                    "method": "upi",
                    "error_reason": "insufficient_funds",
                }
            }
        },
    }
    body = json.dumps(payload, separators=(",", ":")).encode()
    headers = signed_headers(body, "evt_payment_failed_1", secret)

    with TestClient(app) as client:
        first = client.post("/webhooks/razorpay", content=body, headers=headers)
        second = client.post("/webhooks/razorpay", content=body, headers=headers)

    assert first.status_code == 200
    assert first.json()["duplicate"] is False
    assert first.json()["processed"] is True
    assert first.json()["recovery_case_id"]
    assert second.status_code == 200
    assert second.json()["duplicate"] is True

    with get_session() as session:
        events = session.scalars(select(WebhookEvent)).all()
        payments = session.scalars(select(Payment)).all()
        cases = session.scalars(select(RecoveryCase)).all()
        decisions = session.scalars(select(AgentDecision)).all()
        jobs = session.scalars(select(RecoveryJob)).all()

    assert len(events) == 1
    assert events[0].razorpay_event_id == "evt_payment_failed_1"
    assert events[0].processed is True
    assert events[0].delivery_count == 2
    assert len(payments) == 1
    assert payments[0].status == "failed"
    assert payments[0].failure_reason == "insufficient_funds"
    assert len(cases) == 1
    assert cases[0].status == "at_risk"
    assert len(decisions) == 1
    assert decisions[0].execution_mode == "shadow"
    assert len(jobs) == 1
    assert jobs[0].status == "succeeded"


def test_dry_run_policy_decision_is_persisted_idempotently(
    tmp_path,
    monkeypatch,
) -> None:
    secret = configure_test_database(tmp_path, monkeypatch)
    webhook_payload = {
        "event": "payment.failed",
        "payload": {
            "payment": {
                "entity": {
                    "id": "pay_policy_dry_run",
                    "customer_id": "cust_policy",
                    "amount": 125000,
                    "currency": "INR",
                    "method": "upi",
                    "error_reason": "authentication_failed",
                }
            }
        },
    }
    body = json.dumps(webhook_payload, separators=(",", ":")).encode()
    headers = signed_headers(body, "evt_policy_dry_run", secret)

    with TestClient(app) as client:
        webhook = client.post(
            "/webhooks/razorpay",
            content=body,
            headers=headers,
        )
        case_id = webhook.json()["recovery_case_id"]
        decision_payload = {
            "decision_key": "policy-dry-run-001",
            "model_version": "recovery_model_v1",
            "policy_version": "recovery_policy_v3",
            "policy_manifest_sha256": "a" * 64,
            "decision_type": "allow",
            "selected_action": "payment_link",
            "candidate_actions": {
                "payment_link": {
                    "eligible": True,
                    "supported": True,
                }
            },
            "predicted_probabilities": {"payment_link": 0.71},
            "expected_values": {"payment_link": 885.5},
            "decision_reasons": [
                "eligible",
                "highest_risk_adjusted_expected_value",
            ],
            "fallback_used": False,
            "risk_checks": {"passed": True},
            "dry_run": True,
        }
        first = client.post(
            f"/api/recovery/cases/{case_id}/decisions/dry-run",
            json=decision_payload,
        )
        second = client.post(
            f"/api/recovery/cases/{case_id}/decisions/dry-run",
            json=decision_payload,
        )
        listed = client.get(
            f"/api/recovery/cases/{case_id}/decisions"
        )

    assert first.status_code == 200
    assert first.json()["execution_status"] == "would_execute"
    assert first.json()["duplicate"] is False
    assert second.status_code == 200
    assert second.json()["duplicate"] is True
    assert len(listed.json()) == 2
    assert {item["execution_mode"] for item in listed.json()} == {
        "dry_run",
        "shadow",
    }

    with get_session() as session:
        decisions = session.scalars(select(AgentDecision)).all()
        interventions = session.scalars(select(Intervention)).all()
        recovery_case = session.get(RecoveryCase, case_id)

    assert len(decisions) == 2
    assert all(decision.dry_run is True for decision in decisions)
    assert len(interventions) == 1
    assert interventions[0].status == "would_execute"
    assert recovery_case is not None
    assert recovery_case.status == "action_selected_dry_run"
