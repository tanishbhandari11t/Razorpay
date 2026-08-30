from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import yaml

from ml.src.feature_support import (
    automation_eligibility,
    evaluate_feature_support,
    feature_contract,
    model_feature_names,
)
from ml.src.razorpay_failure_taxonomy import classify_razorpay_failure


REPO_ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_phase11_baseline_artifacts_are_unchanged() -> None:
    manifest = yaml.safe_load(
        (
            REPO_ROOT / "ml" / "config" / "phase11_baseline_manifest.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["baseline_id"] == "REAL_RAZORPAY_SHADOW_V1"
    assert manifest["restrictions"]["train_on_cases"] is False
    frozen_inputs = [
        *manifest["artifacts"].values(),
        *manifest["frozen_upstream"].values(),
    ]
    for artifact in frozen_inputs:
        assert _sha256(REPO_ROOT / artifact["path"]) == artifact["sha256"]


def test_deterministic_taxonomy_keeps_generic_failure_unknown() -> None:
    diagnosis = classify_razorpay_failure("payment_failed")
    assert diagnosis.taxonomy == "unknown"
    assert diagnosis.state == "UNKNOWN"
    assert diagnosis.execution_allowed is False
    assert diagnosis.matched_rule == "explicit_unknown:payment_failed"


def test_deterministic_taxonomy_maps_supported_reason() -> None:
    diagnosis = classify_razorpay_failure(
        "international_transaction_not_allowed"
    )
    assert diagnosis.taxonomy == "merchant_configuration"
    assert diagnosis.state == "KNOWN"
    assert diagnosis.confidence == "deterministic"
    assert diagnosis.execution_allowed is False


def test_fraud_flag_always_blocks_execution() -> None:
    diagnosis = classify_razorpay_failure("gateway_timeout", fraud_flag=1)
    assert diagnosis.taxonomy == "fraud_risk"
    assert diagnosis.safe_automation == "never"
    assert diagnosis.execution_allowed is False


def test_online_contract_covers_every_model_input() -> None:
    assert set(feature_contract()) == {
        *model_feature_names(),
        "chosen_intervention",
    }


def test_feature_support_separates_unknown_from_unavailable() -> None:
    features = {name: 1 for name in model_feature_names()}
    result = evaluate_feature_support(features, threshold=0.70)
    assert result.total == 48
    assert result.known == 32
    assert result.unknown == 8
    assert result.unavailable == 8
    assert result.score == pytest.approx(2 / 3)
    assert result.threshold_passed is False


def test_unknown_failure_and_unselected_threshold_fail_closed() -> None:
    features = {name: 1 for name in model_feature_names()}
    support = evaluate_feature_support(features)
    gate = automation_eligibility(
        failure_taxonomy="unknown",
        support=support,
    )
    assert gate["allowed"] is False
    assert gate["execution_allowed"] is False
    assert gate["required_action"] == "escalate_to_merchant"
    assert "unknown_failure_taxonomy" in gate["reasons"]
    assert "feature_support_threshold_not_selected" in gate["reasons"]


def test_phase12_readiness_keeps_controlled_execution_blocked() -> None:
    readiness = json.loads(
        (
            REPO_ROOT
            / "ml"
            / "reports"
            / "phase12"
            / "phase12_readiness.json"
        ).read_text(encoding="utf-8")
    )
    assert readiness["status"] == "execution_blocked"
    assert readiness["execution"]["shadow"] == "ENABLED"
    assert readiness["execution"]["controlled"] == "BLOCKED"
    assert readiness["execution"]["provider_actions_enabled"] is False
