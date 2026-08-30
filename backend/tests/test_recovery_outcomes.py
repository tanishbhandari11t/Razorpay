from __future__ import annotations

import hashlib
import hmac
import json
from datetime import UTC, datetime, timedelta

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.database.connection import get_session, initialize_database
from app.main import app
from app.models.intervention import Intervention
from app.models.intervention_outcome import InterventionOutcome
from app.models.outcome_observation import OutcomeObservation
from app.models.payment import Payment
from app.services.outcome_observation import (
    finalize_due_outcomes,
    observe_pending_outcomes,
    outcome_metrics,
    record_payment_observation,
    snapshot_due_checkpoints,
)
from app.services.outcome_state_machine import mark_outcome_executed, serialize_outcome


def _configure(tmp_path, monkeypatch, name: str) -> str:
    monkeypatch.setenv(
        "DATABASE_URL",
        f"sqlite:///{(tmp_path / name).as_posix()}",
    )
    monkeypatch.setenv("RAZORPAY_KEY_ID", "rzp_test_outcomes")
    monkeypatch.setenv("RAZORPAY_KEY_SECRET", "outcome-key-secret")
    monkeypatch.setenv("RAZORPAY_WEBHOOK_SECRET", "outcome-webhook-secret")
    monkeypatch.setenv("CELERY_TASK_ALWAYS_EAGER", "true")
    monkeypatch.setenv("EXECUTION_MODE", "shadow")
    initialize_database()
    return "outcome-webhook-secret"


def _headers(body: bytes, secret: str, event_id: str) -> dict[str, str]:
    return {
        "Content-Type": "application/json",
        "X-Razorpay-Signature": hmac.new(
            secret.encode(),
            body,
            hashlib.sha256,
        ).hexdigest(),
        "X-Razorpay-Event-Id": event_id,
    }


def _payload(event: str, payment_id: str, status: str) -> bytes:
    payload = {
        "event": event,
        "payload": {
            "payment": {
                "entity": {
                    "id": payment_id,
                    "customer_id": f"cust_{payment_id}",
                    "amount": 249900,
                    "currency": "INR",
                    "method": "upi",
                    "status": status,
                    "error_reason": (
                        "payment_failed" if status == "failed" else None
                    ),
                    "created_at": int(datetime.now(UTC).timestamp()),
                }
            }
        },
    }
    return json.dumps(payload, separators=(",", ":")).encode()


def test_shadow_decision_records_provenance_without_recovery_label(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-shadow.db")
    failed = _payload("payment.failed", "pay_outcome_shadow", "failed")
    captured = _payload(
        "payment.captured",
        "pay_outcome_shadow",
        "captured",
    )
    with TestClient(app) as client:
        failure_response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_outcome_failed"),
        )
        case_id = failure_response.json()["recovery_case_id"]
        before = client.get(
            f"/api/recovery/cases/{case_id}/outcomes"
        ).json()
        capture_response = client.post(
            "/webhooks/razorpay",
            content=captured,
            headers=_headers(captured, secret, "evt_outcome_captured"),
        )
        after = client.get(
            f"/api/recovery/cases/{case_id}/outcomes"
        ).json()
        metrics = client.get("/api/recovery/outcomes/metrics").json()

    assert len(before) == 1
    assert before[0]["outcome_state"] == "decided"
    assert before[0]["attempted"] is False
    assert before[0]["data_source"] == "real_shadow"
    assert capture_response.json()["outcome_observation"] == {
        "observations_inserted": 1,
        "attributed_recoveries": 0,
        "natural_recoveries_observed": 1,
    }
    assert after[0]["outcome_state"] == "decided"
    assert after[0]["payment_recovered"] is None
    assert after[0]["natural_recovery_observed"] is True
    assert after[0]["failure_timestamp"] is not None
    assert after[0]["payment_status_after_24h"] is None
    assert after[0]["label_kind"] == "observational_recovery"
    assert metrics["training_eligible_labels"] == 0
    assert metrics["attributed_intervention_recoveries"] == 0
    assert metrics["observational_recoveries"] == 1
    assert metrics["real_actions_executed"] == 0
    assert metrics["controlled_execution_authorized"] is False


def test_controlled_outcome_window_can_close_without_claiming_nonrecovery(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-timeout.db")
    failed = _payload("payment.failed", "pay_outcome_timeout", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_outcome_timeout"),
        )
    case_id = response.json()["recovery_case_id"]
    attempted_at = datetime.now(UTC)
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        outcome.execution_mode = "controlled"
        outcome.data_source = "real_controlled"
        intervention = Intervention(
            payment_id=outcome.payment_id,
            agent_decision_id=outcome.agent_decision_id,
            type=outcome.action or "payment_link",
            reason="phase14_test_only",
            attempt_number=1,
            status="executed",
            cost=0,
        )
        session.add(intervention)
        session.flush()
        mark_outcome_executed(
            outcome,
            intervention,
            attempted_at=attempted_at,
        )
        outcome.observation_window_ends_at = attempted_at + timedelta(hours=1)
        assert finalize_due_outcomes(
            session,
            now=attempted_at + timedelta(hours=2),
        ) == 1
        session.commit()

    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        observations = session.scalars(
            select(OutcomeObservation)
        ).all()
        assert outcome is not None
        assert outcome.outcome_state == "no_recovery_observed"
        assert outcome.payment_recovered is None
        assert outcome.recovered_amount_minor == 0
        assert observations == []
        assert finalize_due_outcomes(
            session,
            now=attempted_at + timedelta(hours=3),
        ) == 0


def test_attributed_capture_requires_executed_controlled_action(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-recovered.db")
    failed = _payload("payment.failed", "pay_outcome_recovered", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_outcome_recovery_failed"),
        )
    case_id = response.json()["recovery_case_id"]
    attempted_at = datetime.now(UTC)
    observed_at = attempted_at + timedelta(hours=2)
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        outcome.execution_mode = "controlled"
        outcome.data_source = "real_controlled"
        intervention = Intervention(
            payment_id=outcome.payment_id,
            agent_decision_id=outcome.agent_decision_id,
            type=outcome.action or "payment_link",
            reason="phase14_test_only",
            attempt_number=1,
            status="executed",
            cost=0,
        )
        session.add(intervention)
        session.flush()
        mark_outcome_executed(
            outcome,
            intervention,
            attempted_at=attempted_at,
        )
        payment = session.get(Payment, outcome.payment_id)
        assert payment is not None
        payment.status = "captured"
        result = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_attributed_capture",
            observed_at=observed_at,
            payload={"event_type": "payment.captured"},
        )
        session.commit()
    assert result["attributed_recoveries"] == 1
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
    assert outcome is not None
    assert outcome.outcome_state == "recovered"
    assert outcome.payment_recovered is True
    assert outcome.recovered_amount_minor == 249900
    assert outcome.failure_timestamp is not None
    failure_at = outcome.failure_timestamp
    if failure_at.tzinfo is None:
        failure_at = failure_at.replace(tzinfo=UTC)
    assert outcome.time_to_recovery_seconds == int(
        (observed_at - failure_at).total_seconds()
    )
    assert outcome.natural_recovery_observed is False
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        serialized = serialize_outcome(outcome)
        assert serialized["label_kind"] == "attributed_intervention_recovery"


def test_database_checkpoint_records_24h_status_without_attributing_recovery(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-checkpoint.db")
    failed = _payload("payment.failed", "pay_outcome_checkpoint", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_outcome_checkpoint"),
        )
    case_id = response.json()["recovery_case_id"]
    now = datetime.now(UTC)
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        outcome.failure_timestamp = now - timedelta(hours=25)
        outcome.observation_window_starts_at = now - timedelta(hours=25)
        filled = snapshot_due_checkpoints(session, now=now)
        session.commit()
    assert filled["filled_24h"] == 1
    assert filled["filled_48h"] == 0
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
    assert outcome is not None
    assert outcome.payment_status_after_24h == "failed"
    assert outcome.payment_status_after_48h is None
    assert outcome.outcome_state == "decided"
    assert outcome.payment_recovered is None


def _controlled_execute(session, case_id: str, attempted_at: datetime):
    outcome = session.scalar(
        select(InterventionOutcome).where(
            InterventionOutcome.recovery_case_id == case_id
        )
    )
    assert outcome is not None
    outcome.execution_mode = "controlled"
    outcome.data_source = "real_controlled"
    intervention = Intervention(
        payment_id=outcome.payment_id,
        agent_decision_id=outcome.agent_decision_id,
        type=outcome.action or "payment_link",
        reason="phase14_test_only",
        attempt_number=1,
        status="executed",
        cost=0,
    )
    session.add(intervention)
    session.flush()
    mark_outcome_executed(
        outcome,
        intervention,
        attempted_at=attempted_at,
    )
    return outcome


def test_outcome_plumbing_edge_cases_are_idempotent_and_unattributed(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-edges.db")
    failed = _payload("payment.failed", "pay_outcome_edges", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_outcome_edges"),
        )
    case_id = response.json()["recovery_case_id"]
    now = datetime.now(UTC)
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        payment = session.get(Payment, outcome.payment_id)
        assert payment is not None
        payment.status = "captured"
        before = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_before_attempt",
            observed_at=now,
            payload={"event_type": "payment.captured"},
        )
        assert before["attributed_recoveries"] == 0
        assert before["natural_recoveries_observed"] == 1
        assert outcome.outcome_state == "decided"

        attempted_at = now + timedelta(hours=1)
        outcome = _controlled_execute(session, case_id, attempted_at)
        same_ts = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_same_timestamp",
            observed_at=attempted_at,
            payload={"event_type": "payment.captured"},
        )
        duplicate = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_same_timestamp",
            observed_at=attempted_at + timedelta(minutes=1),
            payload={"event_type": "payment.captured"},
        )
        extra = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_second_capture",
            observed_at=attempted_at + timedelta(hours=1),
            payload={"event_type": "payment.captured"},
        )
        session.commit()
    assert same_ts["attributed_recoveries"] == 1
    assert duplicate == {
        "observations_inserted": 0,
        "attributed_recoveries": 0,
        "natural_recoveries_observed": 0,
    }
    assert extra["attributed_recoveries"] == 0

    cancelled = _payload("payment.failed", "pay_outcome_cancel", "failed")
    with TestClient(app) as client:
        cancel_response = client.post(
            "/webhooks/razorpay",
            content=cancelled,
            headers=_headers(cancelled, secret, "evt_outcome_cancel"),
        )
    cancel_id = cancel_response.json()["recovery_case_id"]
    with get_session() as session:
        attempted_at = datetime.now(UTC)
        outcome = _controlled_execute(session, cancel_id, attempted_at)
        payment = session.get(Payment, outcome.payment_id)
        assert payment is not None
        payment.status = "cancelled"
        cancel_obs = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_cancelled",
            observed_at=attempted_at + timedelta(hours=2),
            payload={"event_type": "payment.cancelled"},
        )
        session.commit()
    assert cancel_obs["attributed_recoveries"] == 0
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == cancel_id
            )
        )
    assert outcome is not None
    assert outcome.outcome_state == "waiting_for_outcome"
    assert outcome.payment_recovered is None

    late = _payload("payment.failed", "pay_outcome_late", "failed")
    with TestClient(app) as client:
        late_response = client.post(
            "/webhooks/razorpay",
            content=late,
            headers=_headers(late, secret, "evt_outcome_late"),
        )
    late_id = late_response.json()["recovery_case_id"]
    with get_session() as session:
        attempted_at = datetime.now(UTC)
        outcome = _controlled_execute(session, late_id, attempted_at)
        outcome.observation_window_ends_at = attempted_at + timedelta(hours=1)
        assert finalize_due_outcomes(
            session,
            now=attempted_at + timedelta(hours=2),
        ) == 1
        payment = session.get(Payment, outcome.payment_id)
        assert payment is not None
        payment.status = "captured"
        late_obs = record_payment_observation(
            session,
            payment=payment,
            observation_source="webhook",
            external_ref="evt_late_capture",
            observed_at=attempted_at + timedelta(hours=3),
            payload={"event_type": "payment.captured"},
        )
        session.commit()
    assert late_obs["attributed_recoveries"] == 0
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == late_id
            )
        )
        metrics = outcome_metrics(session)
    assert outcome is not None
    assert outcome.outcome_state == "no_recovery_observed"
    assert outcome.payment_recovered is None
    assert outcome.natural_recovery_observed is True
    assert metrics["attributed_intervention_recoveries"] == 1
    assert metrics["observational_recoveries"] == 1
    assert metrics["training_eligible_labels"] == 1


def test_shadow_observer_records_observational_capture_without_attribution(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-observer-shadow.db")
    failed = _payload("payment.failed", "pay_observer_shadow", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_observer_shadow"),
        )
    case_id = response.json()["recovery_case_id"]
    now = datetime.now(UTC)
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        failure_at = outcome.failure_timestamp or now
        if failure_at.tzinfo is None:
            failure_at = failure_at.replace(tzinfo=UTC)
        payment = session.get(Payment, outcome.payment_id)
        assert payment is not None
        payment.status = "captured"
        capture_at = failure_at + timedelta(hours=6)
        result = observe_pending_outcomes(session, now=capture_at)
        session.commit()
    assert result["inspected"] == 1
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        metrics = outcome_metrics(session)
        serialized = serialize_outcome(outcome)
    assert outcome is not None
    assert outcome.outcome_state == "decided"
    assert outcome.attempted is False
    assert outcome.natural_recovery_observed is True
    assert outcome.payment_recovered is None
    assert outcome.time_to_recovery_seconds == 6 * 3600
    assert serialized["label_kind"] == "observational_recovery"
    assert metrics["attributed_intervention_recoveries"] == 0
    assert metrics["observational_recoveries"] == 1
    assert metrics["training_eligible_labels"] == 0

    # Idempotent second observer pass
    with get_session() as session:
        again = observe_pending_outcomes(session, now=capture_at)
        session.commit()
    assert again["inspected"] == 1


def test_shadow_observer_closes_window_without_capture(
    tmp_path,
    monkeypatch,
) -> None:
    secret = _configure(tmp_path, monkeypatch, "outcome-observer-timeout.db")
    failed = _payload("payment.failed", "pay_observer_timeout", "failed")
    with TestClient(app) as client:
        response = client.post(
            "/webhooks/razorpay",
            content=failed,
            headers=_headers(failed, secret, "evt_observer_timeout"),
        )
    case_id = response.json()["recovery_case_id"]
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
        assert outcome is not None
        ends = outcome.observation_window_ends_at
        if ends.tzinfo is None:
            ends = ends.replace(tzinfo=UTC)
        result = observe_pending_outcomes(
            session,
            now=ends + timedelta(minutes=1),
        )
        session.commit()
    assert result["closed_no_recovery"] == 1
    with get_session() as session:
        outcome = session.scalar(
            select(InterventionOutcome).where(
                InterventionOutcome.recovery_case_id == case_id
            )
        )
    assert outcome is not None
    assert outcome.outcome_state == "no_recovery_observed"
    assert outcome.payment_recovered is None
    assert outcome.natural_recovery_observed is False


