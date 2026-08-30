from __future__ import annotations

from pathlib import Path

from app.services.candidate_audit import build_candidate_audit
from app.services.communication_templates import render_template
from app.services.recoverai_state import (
    RecoverAIState,
    evaluate_recoverai_state,
)
from ml.src.final_evidence_gate import evaluate_final_evidence_gate
from ml.src.generate_final_baseline import build_baseline_manifest
from ml.src.train_final_challenger import write_blocked_model_card


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_final_baseline_includes_frozen_models() -> None:
    baseline = build_baseline_manifest()
    assert baseline["execution_mode"] == "shadow"
    assert baseline["model_ready"] is False
    assert baseline["v1"]["model_hash"]
    assert baseline["v2_online"]["model_hash"]
    assert "ml/config/execution_gate.yaml" in {
        entry["path"] for entry in baseline["artifacts"].values()
    }


def test_recoverai_state_cannot_skip_to_model_ready() -> None:
    progress = evaluate_recoverai_state(
        evidence_records=[],
        challenger_passed=True,
        safety_evaluation_passed=True,
        training_eligible_labels=0,
    )
    assert progress.state in {
        RecoverAIState.OBSERVATION,
        RecoverAIState.EVIDENCE_INSUFFICIENT,
    }
    assert progress.model_ready is False


def test_final_evidence_gate_stays_unauthorized() -> None:
    gate = evaluate_final_evidence_gate([])
    assert gate["phase15_authorized"] is False
    assert gate["candidate_ready"] is False


def test_final_challenger_writes_blocked_model_card(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(REPO_ROOT)
    card = write_blocked_model_card(evidence_records=[])
    assert card["model_ready"] is False
    assert card["status"] == "blocked"
    assert (REPO_ROOT / "ml" / "reports" / "final" / "model_card.json").exists()


def test_candidate_audit_records_all_actions_without_execution() -> None:
    audit = build_candidate_audit(
        payment_id="pay_1",
        predicted_probabilities={
            "retry_payment": 0.61,
            "payment_link": 0.74,
            "whatsapp_reminder": 0.68,
            "escalate_to_merchant": 0.2,
        },
        expected_values={
            "retry_payment": 810,
            "payment_link": 960,
            "whatsapp_reminder": 830,
            "escalate_to_merchant": 350,
        },
        winner="payment_link",
        executed=False,
        blocked_reason="insufficient_real_evidence",
    )
    assert audit["executed"] is False
    assert audit["winner"] == "payment_link"
    assert audit["candidates"]["payment_link"]["probability"] == 0.74


def test_deterministic_template_fallback_exists() -> None:
    message = render_template(
        "payment_link",
        language="hinglish",
        customer_name="Rahul",
        amount_minor=249900,
    )
    assert message["source"] == "deterministic_template"
    assert "2499" in message["message"] or "2,499" in message["message"]
