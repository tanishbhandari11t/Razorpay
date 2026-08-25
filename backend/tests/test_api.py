import hashlib
import hmac
import json

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.connection import get_session, initialize_database
from app.main import app
from app.models.payment import Payment
from app.models.recovery_case import RecoveryCase
from app.models.webhook_event import WebhookEvent


def configure_test_database(tmp_path, monkeypatch) -> str:
    database_url = f"sqlite:///{(tmp_path / 'test.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_unit")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "unit-key-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "unit-webhook-secret")
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

    assert len(events) == 1
    assert events[0].razorpay_event_id == "evt_payment_failed_1"
    assert events[0].processed is True
    assert len(payments) == 1
    assert payments[0].status == "failed"
    assert payments[0].failure_reason == "insufficient_funds"
    assert len(cases) == 1
    assert cases[0].status == "at_risk"
