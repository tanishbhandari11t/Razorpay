from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ml.src.evaluate_policy_sensitivity import (
    perturb_intervention_probabilities,
)
from ml.src.offline_policy_eval import (
    _cluster_bootstrap,
    clipped_propensities,
    doubly_robust_estimate,
    ips_estimate,
)
from ml.src.simulate_recovery import load_config as load_simulator_config


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_ips_handles_low_propensities_safely() -> None:
    rewards = np.array([1.0, 0.0, 1.0])
    target = np.ones(3)
    propensities = np.array([1e-8, 0.5, 0.9])
    ips, snips, weights = ips_estimate(
        rewards,
        target,
        propensities,
        minimum_propensity=0.05,
    )

    assert np.isfinite(ips)
    assert np.isfinite(snips)
    assert weights.max() == 20


def test_propensity_clipping_applies_configured_floor() -> None:
    values = clipped_propensities(
        np.array([0.01, 0.05, 0.9]),
        minimum_propensity=0.05,
    )
    np.testing.assert_array_equal(values, np.array([0.05, 0.05, 0.9]))


def test_doubly_robust_estimator_is_finite() -> None:
    estimate, contributions = doubly_robust_estimate(
        rewards=np.array([1.0, 0.0, 1.0]),
        predicted_logged_reward=np.array([0.8, 0.2, 0.7]),
        predicted_target_reward=np.array([0.7, 0.5, 0.6]),
        target_probability_for_logged_action=np.array([1.0, 0.0, 1.0]),
        logging_propensity=np.array([0.9, 0.9, 0.05]),
        minimum_propensity=0.05,
    )
    assert np.isfinite(estimate)
    assert np.isfinite(contributions).all()


def test_bootstrap_confidence_intervals_are_deterministic() -> None:
    kwargs = {
        "customers": pd.Series(["C1", "C1", "C2", "C3"]),
        "ips_contributions": np.array([1.0, 0.0, 0.5, 1.0]),
        "dr_contributions": np.array([0.8, 0.2, 0.6, 0.9]),
        "weights": np.array([1.0, 1.0, 0.5, 1.0]),
        "weighted_rewards": np.array([1.0, 0.0, 0.5, 1.0]),
        "iterations": 100,
        "confidence_level": 0.95,
        "seed": 42,
    }
    assert _cluster_bootstrap(**kwargs) == _cluster_bootstrap(**kwargs)


def test_policy_decisions_contain_no_counterfactual_outcomes() -> None:
    decisions = pd.read_csv(REPO_ROOT / "ml" / "reports" / "policy_decisions.csv")
    lowered = [column.lower() for column in decisions.columns]
    assert not any("counterfactual" in column for column in lowered)
    assert not any(
        column.endswith("_outcome") or column.endswith("_recovered")
        for column in lowered
        if column != "logged_recovered"
    )


def test_frozen_test_membership_is_unchanged() -> None:
    dataset = pd.read_csv(
        REPO_ROOT
        / "ml"
        / "data"
        / "processed"
        / "logging_policy_dataset.csv"
    )
    decisions = pd.read_csv(REPO_ROOT / "ml" / "reports" / "policy_decisions.csv")
    expected = set(dataset.loc[dataset["split"].eq("test"), "payment_id"])
    assert set(decisions["payment_id"]) == expected
    assert len(decisions) == 1857


def test_probability_multiplier_one_reproduces_simulator_outcomes() -> None:
    outcomes = pd.read_csv(
        REPO_ROOT
        / "ml"
        / "data"
        / "processed"
        / "intervention_outcomes.csv"
    )
    sample = outcomes.iloc[:200].copy()
    reproduced = perturb_intervention_probabilities(
        sample,
        intervention="retry_payment",
        multiplier=1.0,
        seed=int(load_simulator_config()["seed"]),
    )
    np.testing.assert_array_equal(reproduced["recovered"], sample["recovered"])
