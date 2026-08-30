from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "online_feature_schema.yaml"
CANONICAL_SCHEMA_PATH = REPO_ROOT / "ml" / "config" / "feature_schema.yaml"
UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class FeatureState:
    name: str
    state: str
    online_classification: str
    value: Any
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class FeatureSupportResult:
    known: int
    unknown: int
    unavailable: int
    total: int
    score: float
    threshold: float | None
    threshold_passed: bool | None
    features: tuple[FeatureState, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "known": self.known,
            "unknown": self.unknown,
            "unavailable": self.unavailable,
            "total": self.total,
            "score": self.score,
            "threshold": self.threshold,
            "threshold_passed": self.threshold_passed,
            "features": [feature.to_dict() for feature in self.features],
        }


@lru_cache(maxsize=1)
def load_online_feature_schema() -> dict[str, Any]:
    return yaml.safe_load(ONLINE_SCHEMA_PATH.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def model_feature_names() -> tuple[str, ...]:
    schema = yaml.safe_load(CANONICAL_SCHEMA_PATH.read_text(encoding="utf-8"))
    return tuple(
        feature["name"]
        for feature in schema["features"]
        if feature["model_input"]
        and feature["name"] != "chosen_intervention"
    )


@lru_cache(maxsize=1)
def feature_contract() -> dict[str, tuple[str, str]]:
    schema = load_online_feature_schema()
    contract: dict[str, tuple[str, str]] = {}
    deployable = schema["deployable_model_features"]
    for name in deployable["online_available"]:
        contract[name] = ("ONLINE_AVAILABLE", "provider or inference input")
    for name in deployable["online_derivable"]:
        contract[name] = ("ONLINE_DERIVABLE", "strictly prior persisted history")
    excluded = schema["excluded_model_features"]
    for name, reason in excluded["online_unavailable"].items():
        contract[name] = ("ONLINE_UNAVAILABLE", str(reason))
    for name, reason in excluded["semantically_mismatched"].items():
        contract[name] = ("SEMANTICALLY_MISMATCHED", str(reason))
    expected = {*model_feature_names(), "chosen_intervention"}
    if set(contract) != expected:
        missing = sorted(expected - set(contract))
        extra = sorted(set(contract) - expected)
        raise RuntimeError(
            f"Online feature contract mismatch; missing={missing}, extra={extra}"
        )
    return contract


def classify_feature_state(name: str, value: Any) -> FeatureState:
    online_classification, reason = feature_contract()[name]
    if online_classification == "ONLINE_UNAVAILABLE":
        state = "UNAVAILABLE"
    elif online_classification == "SEMANTICALLY_MISMATCHED":
        state = "UNKNOWN"
    elif value is None:
        state = "UNAVAILABLE"
        reason = "Feature value is null"
    elif isinstance(value, str) and value.strip().upper() == UNKNOWN:
        state = "UNKNOWN"
        reason = "Online pipeline emitted the UNKNOWN sentinel"
    else:
        state = "KNOWN"
    return FeatureState(
        name=name,
        state=state,
        online_classification=online_classification,
        value=value,
        reason=reason,
    )


def evaluate_feature_support(
    features: dict[str, Any],
    *,
    threshold: float | None = None,
) -> FeatureSupportResult:
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Feature support threshold must be between 0 and 1")
    states = tuple(
        classify_feature_state(name, features.get(name))
        for name in model_feature_names()
    )
    known = sum(feature.state == "KNOWN" for feature in states)
    unknown = sum(feature.state == "UNKNOWN" for feature in states)
    unavailable = sum(feature.state == "UNAVAILABLE" for feature in states)
    total = len(states)
    score = known / total if total else 0.0
    return FeatureSupportResult(
        known=known,
        unknown=unknown,
        unavailable=unavailable,
        total=total,
        score=score,
        threshold=threshold,
        threshold_passed=(score >= threshold if threshold is not None else None),
        features=states,
    )


def automation_eligibility(
    *,
    failure_taxonomy: str,
    support: FeatureSupportResult,
) -> dict[str, Any]:
    reasons: list[str] = []
    if failure_taxonomy == "unknown":
        reasons.append("unknown_failure_taxonomy")
    if support.threshold is None:
        reasons.append("feature_support_threshold_not_selected")
    elif not support.threshold_passed:
        reasons.append("feature_support_below_threshold")
    if support.unavailable:
        reasons.append("online_unavailable_features_present")
    return {
        "allowed": not reasons,
        "reasons": reasons,
        "required_action": "escalate_to_merchant" if reasons else None,
        "execution_allowed": False if reasons else None,
    }
