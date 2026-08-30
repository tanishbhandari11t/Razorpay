from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH = REPO_ROOT / "ml" / "config" / "policy_evaluation.yaml"


def load_support_policy_config(
    path: Path = DEFAULT_CONFIG_PATH,
) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def context_buckets(
    row: Mapping[str, Any],
    config: dict[str, Any],
) -> tuple[str, str, str]:
    support = config["support"]
    history_boundaries = support["history_success_boundaries"]
    success_rate = float(row["historical_success_rate"])
    if int(row["has_prior_history"]) == 0:
        history = "no_history"
    elif success_rate <= float(history_boundaries[0]):
        history = "lower_history"
    elif success_rate < float(history_boundaries[1]):
        history = "upper_imperfect_history"
    else:
        history = "perfect_history"

    amount = float(row["amount_inr"])
    amount_boundaries = support["amount_boundaries_inr"]
    if amount <= float(amount_boundaries[0]):
        amount_bucket = "amount_q1"
    elif amount <= float(amount_boundaries[1]):
        amount_bucket = "amount_q2"
    elif amount <= float(amount_boundaries[2]):
        amount_bucket = "amount_q3"
    else:
        amount_bucket = "amount_q4"

    failures = int(row["failures_last_30d"])
    if failures == 0:
        failure_bucket = "failures_0"
    elif failures == 1:
        failure_bucket = "failures_1"
    else:
        failure_bucket = "failures_2_plus"
    return history, amount_bucket, failure_bucket


def add_context_buckets(
    dataframe: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = dataframe.copy()
    buckets = result.apply(
        lambda row: context_buckets(row, config),
        axis=1,
        result_type="expand",
    )
    buckets.columns = [
        "history_bucket",
        "amount_bucket",
        "failure_bucket",
    ]
    return pd.concat([result, buckets], axis=1)


def _effective_sample_size(propensities: pd.Series) -> float:
    weights = 1 / propensities.astype(float).to_numpy()
    return float(weights.sum() ** 2 / np.square(weights).sum())


def build_context_support_table(
    training_data: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    bucketed = add_context_buckets(training_data, config)
    eligible = bucketed.loc[
        bucketed["chosen_intervention"].isin(config["interventions"])
    ]
    table = (
        eligible.groupby(
            [
                "history_bucket",
                "amount_bucket",
                "failure_bucket",
                "chosen_intervention",
            ],
            observed=True,
        )
        .agg(
            action_count=("payment_id", "size"),
            effective_sample_size=(
                "policy_probability",
                _effective_sample_size,
            ),
            observed_recovery_rate=("recovered", "mean"),
        )
        .reset_index()
        .rename(columns={"chosen_intervention": "action"})
    )
    table["effective_sample_size"] = table["effective_sample_size"].round(8)
    table["observed_recovery_rate"] = table["observed_recovery_rate"].round(8)
    minimum_count = int(config["support"]["min_context_action_count"])
    minimum_ess = float(
        config["support"]["min_context_effective_sample_size"]
    )
    table["supported"] = (
        table["action_count"].ge(minimum_count)
        & table["effective_sample_size"].ge(minimum_ess)
    )
    return table


@dataclass(frozen=True)
class SupportEvidence:
    action: str
    action_count: int
    effective_sample_size: float
    supported: bool
    context: tuple[str, str, str]


class SupportIndex:
    def __init__(
        self,
        table: pd.DataFrame,
        config: dict[str, Any],
    ) -> None:
        self.config = config
        self._lookup = {
            (
                str(row.history_bucket),
                str(row.amount_bucket),
                str(row.failure_bucket),
                str(row.action),
            ): (int(row.action_count), float(row.effective_sample_size))
            for row in table.itertuples(index=False)
        }

    @classmethod
    def from_training_data(
        cls,
        training_data: pd.DataFrame,
        config: dict[str, Any],
    ) -> "SupportIndex":
        return cls(build_context_support_table(training_data, config), config)

    def evidence(
        self,
        row: Mapping[str, Any],
        action: str,
    ) -> SupportEvidence:
        context = context_buckets(row, self.config)
        count, ess = self._lookup.get((*context, action), (0, 0.0))
        supported = (
            count >= int(self.config["support"]["min_context_action_count"])
            and ess
            >= float(
                self.config["support"]["min_context_effective_sample_size"]
            )
        )
        return SupportEvidence(action, count, ess, supported, context)


@dataclass(frozen=True)
class SupportSafeDecision:
    selected_action: str
    raw_best_action: str
    selected_probability: float
    decision_margin: float | None
    fallback_used: bool
    decision_reason: str
    selected_action_supported: bool
    candidate_evidence: dict[str, SupportEvidence]

    def candidate_summary_json(
        self,
        probabilities: Mapping[str, float],
    ) -> str:
        return json.dumps(
            {
                action: {
                    "probability": round(float(probabilities[action]), 8),
                    "action_count": evidence.action_count,
                    "effective_sample_size": round(
                        evidence.effective_sample_size,
                        8,
                    ),
                    "supported": evidence.supported,
                }
                for action, evidence in self.candidate_evidence.items()
            },
            sort_keys=True,
        )


def select_support_safe_action(
    row: Mapping[str, Any],
    probabilities: Mapping[str, float],
    support_index: SupportIndex,
    config: dict[str, Any],
) -> SupportSafeDecision:
    interventions = [str(value) for value in config["interventions"]]
    if set(probabilities) != set(interventions):
        raise ValueError("Candidate probabilities do not match interventions")
    if not all(0 <= float(value) <= 1 for value in probabilities.values()):
        raise ValueError("Candidate probabilities must be in [0, 1]")

    ranked_all = sorted(
        interventions,
        key=lambda action: (-float(probabilities[action]), interventions.index(action)),
    )
    raw_best = ranked_all[0]
    evidence = {
        action: support_index.evidence(row, action)
        for action in interventions
    }
    if int(row["fraud_flag"]) == 1:
        return SupportSafeDecision(
            selected_action=str(config["decision"]["fraud_action"]),
            raw_best_action=raw_best,
            selected_probability=0.0,
            decision_margin=None,
            fallback_used=True,
            decision_reason="fraud_policy_block",
            selected_action_supported=False,
            candidate_evidence=evidence,
        )

    supported = [
        action for action in ranked_all if evidence[action].supported
    ]
    if not supported:
        fallback = str(
            config["decision"]["no_supported_action_fallback"]
        )
        return SupportSafeDecision(
            selected_action=fallback,
            raw_best_action=raw_best,
            selected_probability=float(probabilities[fallback]),
            decision_margin=None,
            fallback_used=True,
            decision_reason="no_supported_candidate_safety_fallback",
            selected_action_supported=False,
            candidate_evidence=evidence,
        )

    preferred_fallback = str(
        config["decision"]["preferred_fallback_action"]
    )
    if len(supported) == 1:
        selected = (
            preferred_fallback
            if preferred_fallback in supported
            else supported[0]
        )
        return SupportSafeDecision(
            selected_action=selected,
            raw_best_action=raw_best,
            selected_probability=float(probabilities[selected]),
            decision_margin=None,
            fallback_used=True,
            decision_reason="single_supported_candidate",
            selected_action_supported=True,
            candidate_evidence=evidence,
        )

    margin = float(probabilities[supported[0]]) - float(
        probabilities[supported[1]]
    )
    minimum_margin = float(
        config["decision"]["minimum_probability_margin"]
    )
    if margin >= minimum_margin:
        selected = supported[0]
        fallback_used = False
        reason = "highest_supported_probability"
    else:
        selected = (
            preferred_fallback
            if preferred_fallback in supported
            else supported[0]
        )
        fallback_used = True
        reason = "insufficient_probability_margin"

    return SupportSafeDecision(
        selected_action=selected,
        raw_best_action=raw_best,
        selected_probability=float(probabilities[selected]),
        decision_margin=margin,
        fallback_used=fallback_used,
        decision_reason=reason,
        selected_action_supported=evidence[selected].supported,
        candidate_evidence=evidence,
    )
